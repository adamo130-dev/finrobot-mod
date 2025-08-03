import os, json, time, random
from collections import defaultdict
from datetime import date, datetime, timedelta
import gradio as gr

import pandas as pd
import finnhub
from openai import OpenAI

from io import StringIO
import requests


# ---------- 0  CONFIG ---------------------------------------------------------

def get_openai_api_key(model_name="gpt-4o-mini", config_path="OAI_CONFIG_LIST"):
    with open(config_path, "r") as f:
        configs = json.load(f)
    for entry in configs:
        if entry.get("model") == model_name:
            return entry.get("api_key")
    raise RuntimeError(f"API key for model {model_name} not found in {config_path}")

OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = get_openai_api_key(OPENAI_MODEL)
client = OpenAI(api_key=OPENAI_API_KEY)

def load_api_keys(config_path="../finrobot-mod/config_api_keys"):
    with open(config_path, "r") as f:
        return json.load(f)

api_keys = load_api_keys()

FINNHUB_KEY = api_keys.get("finnhub_api_key") or api_keys.get("FINNHUB_API_KEY")
ALPHA_KEY = os.getenv("ALPHA_KEY")  # If you want to support env override
if not ALPHA_KEY:
    ALPHA_KEY = "QKAWCJ6Q19ZYBB3S"  # fallback or load from config if you add it there

finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)


SYSTEM_PROMPT = (
    "You are a seasoned stock-market analyst. "
    "Given recent company news and optional basic financials, "
    "return:\n"
    "[Positive Developments] – 2-4 bullets\n"
    "[Potential Concerns] – 2-4 bullets\n"
    "[Prediction & Analysis] – a one-week price outlook with rationale."
)


# ---------- 1  DATE / UTILITY HELPERS ----------------------------------------

def today() -> str:
    return date.today().strftime("%Y-%m-%d")

def n_weeks_before(date_string: str, n: int) -> str:
    return (datetime.strptime(date_string, "%Y-%m-%d") -
            timedelta(days=7 * n)).strftime("%Y-%m-%d")


# ---------- 2  DATA FETCHING --------------------------------------------------

def get_stock_data(symbol: str, steps: list[str]) -> pd.DataFrame:

    if not ALPHA_KEY:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is Missing")

    # 免费端点：TIME_SERIES_DAILY :contentReference[oaicite:8]{index=8}
    url = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}"
        f"&apikey={ALPHA_KEY}"
        "&datatype=csv"
        "&outputsize=full"
    )

    # 重试 3 次
    text = None
    for attempt in range(3):
        resp = requests.get(url, timeout=10)
        if not resp.ok:
            time.sleep(1)
            continue
        text = resp.text.strip()
        if text.startswith("{"):
            info = resp.json()
            msg = info.get("Note") or info.get("Error Message") or str(info)
            raise RuntimeError(f"Alpha Vantage Return Error：{msg}")
        break

    if not text:
        raise RuntimeError(f"Alpha Vantage Connection Error：{url}")

    df = pd.read_csv(StringIO(text))
    date_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    data = {"Start Date": [], "End Date": [], "Start Price": [], "End Price": []}
    for i in range(len(steps) - 1):
        s_date = pd.to_datetime(steps[i])
        e_date = pd.to_datetime(steps[i+1])
        seg = df.loc[s_date:e_date]
        if seg.empty:
            raise RuntimeError(
                f"Alpha Vantage 无法获取 {symbol} 在 {steps[i]} – {steps[i+1]} 的数据"
            )
        data["Start Date"].append(seg.index[0])
        data["Start Price"].append(seg["close"].iloc[0])
        data["End Date"].append(seg.index[-1])
        data["End Price"].append(seg["close"].iloc[-1])
        # Limits：5 times/min
        time.sleep(12)

    return pd.DataFrame(data)

def current_basics(symbol: str, curday: str) -> dict:
    raw = finnhub_client.company_basic_financials(symbol, "all")
    if not raw["series"]:
        return {}
    merged = defaultdict(dict)
    for metric, vals in raw["series"]["quarterly"].items():
        for v in vals:
            merged[v["period"]][metric] = v["v"]

    latest = max((p for p in merged if p <= curday), default=None)
    if latest is None:
        return {}
    d = dict(merged[latest])
    d["period"] = latest
    return d

def attach_news(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    news_col = []
    for _, row in df.iterrows():
        start = row["Start Date"].strftime("%Y-%m-%d")
        end   = row["End Date"].strftime("%Y-%m-%d")
        time.sleep(1)                                        # Finnhub QPM guard
        weekly = finnhub_client.company_news(symbol, _from=start, to=end)
        weekly_fmt = [
            {
                "date"    : datetime.fromtimestamp(n["datetime"]).strftime("%Y%m%d%H%M%S"),
                "headline": n["headline"],
                "summary" : n["summary"],
            }
            for n in weekly
        ]
        weekly_fmt.sort(key=lambda x: x["date"])
        news_col.append(json.dumps(weekly_fmt))
    df["News"] = news_col
    return df


# ---------- 3  PROMPT CONSTRUCTION -------------------------------------------

def sample_news(news: list[str], k: int = 5) -> list[str]:
    if len(news) <= k: return news
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]


def make_prompt(symbol: str, df: pd.DataFrame, curday: str, use_basics=False) -> str:
    # Company profile
    prof = finnhub_client.company_profile2(symbol=symbol)
    company_blurb = (
        f"[Company Introduction]:\n{prof['name']} operates in the "
        f"{prof['finnhubIndustry']} sector ({prof['country']}). "
        f"Founded {prof['ipo']}, market cap {prof['marketCapitalization']:.1f} "
        f"{prof['currency']}; ticker {symbol} on {prof['exchange']}.\n"
    )

    # Past weeks block
    past_block = ""
    for _, row in df.iterrows():
        term = "increased" if row["End Price"] > row["Start Price"] else "decreased"
        head = (f"From {row['Start Date']:%Y-%m-%d} to {row['End Date']:%Y-%m-%d}, "
                f"{symbol}'s stock price {term} from "
                f"{row['Start Price']:.2f} to {row['End Price']:.2f}.")
        news_items = json.loads(row["News"])
        summaries  = [
            f"[Headline] {n['headline']}\n[Summary] {n['summary']}\n"
            for n in news_items
            if not n["summary"].startswith("Looking for stock market analysis")
        ]
        past_block += "\n" + head + "\n" + "".join(sample_news(summaries, 5))

    # Optional basic financials
    if use_basics:
        basics = current_basics(symbol, curday)
        if basics:
            basics_txt = "\n".join(f"{k}: {v}" for k, v in basics.items() if k != "period")
            basics_block = (f"\n[Basic Financials] (reported {basics['period']}):\n{basics_txt}\n")
        else:
            basics_block = "\n[Basic Financials]: not available\n"
    else:
        basics_block = "\n[Basic Financials]: not requested\n"

    horizon = f"{curday} to {n_weeks_before(curday, -1)}"
    final_user_msg = (
        company_blurb
        + past_block
        + basics_block
        + f"\nBased on all information before {curday}, analyse positive "
          "developments and potential concerns for {symbol}, then predict its "
          f"price movement for next week ({horizon})."
    )
    return final_user_msg


# ---------- 4  LLM CALL -------------------------------------------------------

def chat_completion(prompt: str,
                    model: str = OPENAI_MODEL,
                    temperature: float = 0.3,
                    stream: bool = False) -> str:

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        stream=stream,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
    )

    if stream:
        collected = []
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            collected.append(delta)
        print()
        return "".join(collected)

    # without stream
    return response.choices[0].message.content


# ---------- 5  MAIN ENTRY (CLI test) -----------------------------------------

def predict(symbol: str = "AAPL",
            curday: str = today(),
            n_weeks: int = 3,
            use_basics: bool = False,
            stream: bool = False) -> tuple[str, str]:
    steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]
    df    = get_stock_data(symbol, steps)
    df    = attach_news(symbol, df)

    prompt_info = make_prompt(symbol, df, curday, use_basics)
    answer      = chat_completion(prompt_info, stream=stream)

    return prompt_info, answer



# ---------- 6  SETUP HF -----------------------------------------


def hf_predict(symbol, n_weeks, use_basics):
    # 1. get curday
    curday = date.today().strftime("%Y-%m-%d")
    # 2. call predict
    prompt, answer = predict(
        symbol=symbol.upper(),
        curday=curday,
        n_weeks=int(n_weeks),
        use_basics=bool(use_basics),
        stream=False
    )
    return prompt, answer

with gr.Blocks() as demo:
    gr.Markdown("FinRobot_Forecaster")
    with gr.Row():
        symbol = gr.Textbox(label="Ticker（eg. AAPL）", value="AAPL")
        n_weeks = gr.Slider(1, 6, value=3, step=1, label="Trace Back Weeks")
        use_basics = gr.Checkbox(label="Add Basic Financials", value=False)
    output_prompt = gr.Textbox(label="Model Prompt", lines=8)
    output_answer = gr.Textbox(label="Model Output", lines=12)
    btn = gr.Button("Run Forecaster")
    btn.click(fn=hf_predict,
              inputs=[symbol, n_weeks, use_basics],
              outputs=[output_prompt, output_answer])

if __name__ == "__main__":
    demo.launch()
