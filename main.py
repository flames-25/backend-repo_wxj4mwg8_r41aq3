import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import requests

from database import db, create_document

app = FastAPI(title="US Trading Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Models --------------------
class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="US ticker symbol, e.g., AAPL")
    timeframe: str = Field("1d", description="Yahoo interval: 1d,1h,30m,15m,5m")
    lookback_days: int = Field(200, ge=30, le=2000)

class AnalyzeResponse(BaseModel):
    symbol: str
    last_price: Optional[float]
    trend: str
    rsi: Optional[float]
    sma_50: Optional[float]
    sma_200: Optional[float]
    signal: str
    reason: str

class PaperTradeRequest(BaseModel):
    symbol: str
    side: str = Field(..., pattern="^(buy|sell)$")
    quantity: float = Field(..., gt=0)

class BrokerLogin(BaseModel):
    broker: str = Field(..., description="Supported: alpaca")
    api_key: str
    api_secret: str
    paper: bool = True

class PlaceOrder(BaseModel):
    symbol: str
    side: str = Field(..., pattern="^(buy|sell)$")
    qty: float
    type: str = Field("market", pattern="^(market|limit)$")
    time_in_force: str = Field("day")
    limit_price: Optional[float] = None

# -------------------- Helpers --------------------

def _yahoo_range(days: int) -> str:
    # Yahoo supports ranges like 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    if days <= 5:
        return f"{days}d"
    if days <= 30:
        return "1mo"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    if days <= 1825:
        return "5y"
    return "10y"


def fetch_prices(symbol: str, interval: str, lookback_days: int) -> List[float]:
    base = "https://query1.finance.yahoo.com/v8/finance/chart/{}".format(symbol)
    params = {
        "interval": interval,
        "range": _yahoo_range(lookback_days),
        "includePrePost": "false",
        "events": "div,splits",
    }
    r = requests.get(base, params=params, timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Yahoo fetch failed: {r.text[:120]}")
    j = r.json()
    try:
        closes = j["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    except Exception:
        raise HTTPException(status_code=404, detail="No data for symbol/timeframe")
    if not closes:
        raise HTTPException(status_code=404, detail="Empty data series")
    # filter out None values occasionally present in Yahoo data
    prices = [float(c) for c in closes if c is not None]
    if len(prices) < 50:
        raise HTTPException(status_code=404, detail="Insufficient data returned")
    return prices


def sma(values: List[float], window: int) -> Optional[float]:
    if len(values) < window:
        return None
    s = sum(values[-window:])
    return s / window


def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(-period, 0):
        change = values[i] - values[i - 1]
        if change > 0:
            gains.append(change)
        else:
            losses.append(-change)
    avg_gain = (sum(gains) / period) if gains else 0.0
    avg_loss = (sum(losses) / period) if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -------------------- Routes --------------------
@app.get("/")
def read_root():
    return {"message": "US Trading Assistant Backend running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_name"] = getattr(db, 'name', None)
            try:
                response["collections"] = db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    symbol = req.symbol.upper().strip()
    prices = fetch_prices(symbol, req.timeframe, req.lookback_days)

    last_price = prices[-1]
    s50 = sma(prices, 50)
    s200 = sma(prices, 200)
    r = rsi(prices, 14)

    trend = "bullish" if (s50 is not None and s200 is not None and s50 > s200) else "bearish"

    signal = "hold"
    reasons: List[str] = []

    if r is not None:
        if r < 30:
            signal = "buy"; reasons.append("RSI oversold (<30)")
        elif r > 70:
            signal = "sell"; reasons.append("RSI overbought (>70)")

    if s50 is not None and s200 is not None:
        reasons.append("Golden cross bias" if s50 > s200 else "Death cross bias")

    if signal == "hold" and s50 is not None:
        if trend == "bullish" and last_price > s50:
            signal = "buy"; reasons.append("Price above SMA50 in bullish regime")
        elif trend == "bearish" and last_price < s50:
            signal = "sell"; reasons.append("Price below SMA50 in bearish regime")

    reason = "; ".join(reasons) if reasons else "Neutral conditions"

    try:
        create_document("signal", {
            "symbol": symbol,
            "timeframe": req.timeframe,
            "signal": signal,
            "reason": reason,
            "price": last_price,
            "generated_at": datetime.utcnow().isoformat()
        })
    except Exception:
        pass

    return AnalyzeResponse(
        symbol=symbol,
        last_price=round(last_price, 4) if last_price is not None else None,
        trend=trend,
        rsi=round(r, 2) if r is not None else None,
        sma_50=round(s50, 2) if s50 is not None else None,
        sma_200=round(s200, 2) if s200 is not None else None,
        signal=signal,
        reason=reason,
    )

@app.post("/api/paper-trade")
def paper_trade(order: PaperTradeRequest):
    doc_id = None
    try:
        doc_id = create_document("papertrade", {
            "symbol": order.symbol.upper(),
            "side": order.side,
            "quantity": order.quantity,
            "status": "filled",
            "filled_price": None,
            "ts": datetime.utcnow().isoformat()
        })
    except Exception:
        pass
    return {"status": "simulated", "order_id": doc_id}

@app.post("/api/broker/test")
def broker_test(creds: BrokerLogin):
    if creds.broker.lower() != "alpaca":
        raise HTTPException(status_code=400, detail="Only Alpaca is supported in this demo")
    base = "https://paper-api.alpaca.markets" if creds.paper else "https://api.alpaca.markets"
    r = requests.get(f"{base}/v2/account", headers={
        "APCA-API-KEY-ID": creds.api_key,
        "APCA-API-SECRET-KEY": creds.api_secret,
    }, timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    acct = r.json()
    return {"status": "ok", "account_id": acct.get("id"), "cash": acct.get("cash")}

@app.post("/api/broker/order")
def broker_order(creds: BrokerLogin, order: PlaceOrder):
    if creds.broker.lower() != "alpaca":
        raise HTTPException(status_code=400, detail="Only Alpaca is supported in this demo")
    base = "https://paper-api.alpaca.markets" if creds.paper else "https://api.alpaca.markets"
    payload = {
        "symbol": order.symbol.upper(),
        "side": order.side,
        "qty": order.qty,
        "type": order.type,
        "time_in_force": order.time_in_force,
    }
    if order.type == "limit" and order.limit_price is not None:
        payload["limit_price"] = order.limit_price
    r = requests.post(f"{base}/v2/orders", headers={
        "APCA-API-KEY-ID": creds.api_key,
        "APCA-API-SECRET-KEY": creds.api_secret,
        "Content-Type": "application/json"
    }, json=payload, timeout=20)
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
