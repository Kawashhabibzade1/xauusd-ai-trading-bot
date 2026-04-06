//+------------------------------------------------------------------+
//|                                               Features_Other.mqh |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property version   "1.0.0"

//--- CHANGELOG
// v1.0.0: Time (6), Volatility (8), Price Action (6), Sentiment (1) + SMC Quality Score (1).
//         Matching src/feature_engineering.py Lines 141-259.

class COtherFeatures
  {
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;
   int               m_hATR14;   // Handle für ATR14 (wird von Volatility mehrfach benötigt)

public:
                     COtherFeatures() {}
                    ~COtherFeatures() { if(m_hATR14 != INVALID_HANDLE) IndicatorRelease(m_hATR14); }
   
   bool              Init(string symbol, ENUM_TIMEFRAMES tf);
   bool              ComputeTime(int shift, int &index, float &features_array[]);
   bool              ComputeVolatility(int shift, int &index, float &features_array[], double atr14);
   bool              ComputePriceAction(int shift, int &index, float &features_array[], double atr14);
   bool              ComputeSMC(int shift, int &index, float &features_array[],
                                double liquidity_sweep_high, double liquidity_sweep_low,
                                double fvg_size, double bullish_ob, double bearish_ob);
  };

bool COtherFeatures::Init(string symbol, ENUM_TIMEFRAMES tf)
  {
   m_symbol    = symbol;
   m_timeframe = tf;
   m_hATR14    = iATR(m_symbol, m_timeframe, 14);
   return (m_hATR14 != INVALID_HANDLE);
  }

//--- BLOCK 1: TIME FEATURES (6)
bool COtherFeatures::ComputeTime(int shift, int &index, float &features_array[])
  {
   MqlDateTime dt;
   TimeToStruct(iTime(m_symbol, m_timeframe, shift), dt);
   
   int hour   = dt.hour;
   int minute = dt.min;
   int dow    = dt.day_of_week;
   
   // London open: 08:00 UTC => 8*60 = 480 Minuten seit Mitternacht
   int london_open_minute = 8 * 60;
   int ny_open_minute     = 13 * 60 + 30; // 13:30 UTC

   int min_since_midnight = hour * 60 + minute;
   int min_since_london   = min_since_midnight - london_open_minute;
   int min_since_ny       = min_since_midnight - ny_open_minute;
   
   // Session Position: 0=NY-Eröffnung, 1=NY-Close (nach 3.5h = 210 Min)
   double session_pos = (double)min_since_ny / (3.5 * 60.0);
   session_pos = MathMax(0.0, MathMin(1.0, session_pos));

   features_array[index++] = (float)hour;
   features_array[index++] = (float)minute;
   features_array[index++] = (float)dow;
   features_array[index++] = (float)min_since_london;
   features_array[index++] = (float)min_since_ny;
   features_array[index++] = (float)session_pos;
   return true;
  }

//--- BLOCK 2: VOLATILITY FEATURES (8)
bool COtherFeatures::ComputeVolatility(int shift, int &index, float &features_array[], double atr14)
  {
   double close_0 = iClose(m_symbol, m_timeframe, shift);
   double high_0  = iHigh(m_symbol,  m_timeframe, shift);
   double low_0   = iLow(m_symbol,   m_timeframe, shift);
   double high_1  = iHigh(m_symbol,  m_timeframe, shift+1);
   double low_1   = iLow(m_symbol,   m_timeframe, shift+1);
   
   //--- ATR Percentile (Python: rolling(240).apply percentile)
   // Wir berechnen: Wie viele der letzten 240 ATR-Werte sind KLEINER als der aktuelle ATR?
   double buf[1];
   int below = 0;
   for(int i=0; i<240; i++)
     {
      double a[1];
      if(CopyBuffer(m_hATR14, 0, shift+i, 1, a) > 0)
         if(a[0] <= atr14) below++;
     }
   double atr_percentile = (double)below / 240.0;

   //--- Tick Volatility (rolling std über 10 Kerzen der Close-Preise)
   double mean_close = 0;
   for(int i=0; i<10; i++) mean_close += iClose(m_symbol, m_timeframe, shift+i);
   mean_close /= 10.0;
   double var_close = 0;
   for(int i=0; i<10; i++) var_close += MathPow(iClose(m_symbol, m_timeframe, shift+i) - mean_close, 2);
   double tick_volatility = MathSqrt(var_close / 10.0);

   //--- Range Expansion = (H-L) / (H1-L1)
   double denom_rng = (high_1 - low_1);
   double range_expansion = (denom_rng > 0) ? ((high_0 - low_0) / denom_rng) : 1.0;

   //--- Volatility Regime = (ATR14 - ATR14_mean) / ATR14_std über 240 Kerzen
   double atr_mean = 0, atr_std = 0;
   double atr_vals[240];
   for(int i=0; i<240; i++)
     {
      double a[1]; CopyBuffer(m_hATR14, 0, shift+i, 1, a);
      atr_vals[i] = a[0]; atr_mean += a[0];
     }
   atr_mean /= 240.0;
   for(int i=0; i<240; i++) atr_std += MathPow(atr_vals[i] - atr_mean, 2);
   atr_std = MathSqrt(atr_std / 240.0);
   double vol_regime = (atr_std > 0) ? ((atr14 - atr_mean) / atr_std) : 0.0;

   //--- True Range
   double true_range = high_0 - low_0;

   //--- TR Percentile (über 60 Kerzen)
   int tr_below = 0;
   for(int i=0; i<60; i++)
     {
      double h = iHigh(m_symbol,m_timeframe,shift+i);
      double l = iLow(m_symbol, m_timeframe,shift+i);
      if((h-l) <= true_range) tr_below++;
     }
   double tr_percentile = (double)tr_below / 60.0;

   //--- Price Velocity (close.diff(3) / 3) und Acceleration (velocity.diff())
   double close_3 = iClose(m_symbol, m_timeframe, shift+3);
   double price_velocity = (close_0 - close_3) / 3.0;
   
   double close_4 = iClose(m_symbol, m_timeframe, shift+4);
   double prev_velocity = (iClose(m_symbol,m_timeframe,shift+1) - close_4) / 3.0;
   double price_acceleration = price_velocity - prev_velocity;

   features_array[index++] = (float)atr_percentile;
   features_array[index++] = (float)tick_volatility;
   features_array[index++] = (float)range_expansion;
   features_array[index++] = (float)vol_regime;
   features_array[index++] = (float)true_range;
   features_array[index++] = (float)tr_percentile;
   features_array[index++] = (float)price_velocity;
   features_array[index++] = (float)price_acceleration;
   return true;
  }

//--- BLOCK 3: PRICE ACTION (6) + SENTIMENT (1) = 7 features
bool COtherFeatures::ComputePriceAction(int shift, int &index, float &features_array[], double atr14)
  {
   double close_0  = iClose(m_symbol, m_timeframe, shift);
   double close_1  = iClose(m_symbol, m_timeframe, shift+1);
   double close_5  = iClose(m_symbol, m_timeframe, shift+5);
   double close_15 = iClose(m_symbol, m_timeframe, shift+15);
   double close_14 = iClose(m_symbol, m_timeframe, shift+14);
   
   double returns_1m  = (close_1  > 0) ? ((close_0 - close_1)  / close_1)  : 0.0;
   double returns_5m  = (close_5  > 0) ? ((close_0 - close_5)  / close_5)  : 0.0;
   double returns_15m = (close_15 > 0) ? ((close_0 - close_15) / close_15) : 0.0;
   
   double momentum = close_0 - close_14;

   //--- Rolling 50 High/Low für Distanz-Features
   double high50 = iHigh(m_symbol, m_timeframe, shift);
   double low50  = iLow(m_symbol,  m_timeframe, shift);
   for(int i=1; i<50; i++)
     {
      high50 = MathMax(high50, iHigh(m_symbol, m_timeframe, shift+i));
      low50  = MathMin(low50,  iLow(m_symbol,  m_timeframe, shift+i));
     }
   double dist_to_high = (atr14 > 0) ? ((high50 - close_0) / atr14) : 0.0;
   double dist_to_low  = (atr14 > 0) ? ((close_0 - low50)  / atr14) : 0.0;

   features_array[index++] = (float)returns_1m;
   features_array[index++] = (float)returns_5m;
   features_array[index++] = (float)returns_15m;
   features_array[index++] = (float)momentum;
   features_array[index++] = (float)dist_to_high;
   features_array[index++] = (float)dist_to_low;
   features_array[index++] = 0.0f; // Sentiment placeholder (Python: df['sentiment'] = 0.0)
   return true;
  }

//--- BLOCK 4: SMC QUALITY SCORE (1 feature)
bool COtherFeatures::ComputeSMC(int shift, int &index, float &features_array[],
                                double liquidity_sweep_high, double liquidity_sweep_low,
                                double fvg_size, double bullish_ob, double bearish_ob)
  {
   double close_0   = iClose(m_symbol, m_timeframe, shift);
   double close_240 = iClose(m_symbol, m_timeframe, shift+240);

   //--- H4 Range (240 M1 Kerzen = 4 Stunden)
   double h4_high = iHigh(m_symbol, m_timeframe, shift);
   double h4_low  = iLow(m_symbol,  m_timeframe, shift);
   for(int i=1; i<240; i++)
     {
      h4_high = MathMax(h4_high, iHigh(m_symbol, m_timeframe, shift+i));
      h4_low  = MathMin(h4_low,  iLow(m_symbol,  m_timeframe, shift+i));
     }
   double h4_mid = (h4_high + h4_low) / 2.0;

   //--- Step 1: H4 Directional Bias
   // Bullisch: Preis über Mitte UND Close höher als vor 240 Minuten
   int h4_bias = 0;
   if(close_0 > h4_mid && close_0 > close_240)       h4_bias = 1;
   else if(close_0 < h4_mid && close_0 < close_240)  h4_bias = -1;

   //--- Step 2: Premium / Discount Zone
   int in_discount = (close_0 < h4_mid) ? 1 : 0;
   int in_premium  = (close_0 > h4_mid) ? 1 : 0;

   //--- Step 3: Inducement (Liquidity Sweep wurde genommen)
   int inducement = ((liquidity_sweep_high > 0.5) || (liquidity_sweep_low > 0.5)) ? 1 : 0;

   //--- Step 4: Entry Zone vorhanden (FVG oder Order Block)
   int entry_zone = ((fvg_size > 0) || (bullish_ob > 0.5) || (bearish_ob > 0.5)) ? 1 : 0;

   //--- SMC Quality Score: 0-4 Punkte
   int smc_score = (h4_bias != 0 ? 1 : 0) +
                   ((in_discount || in_premium) ? 1 : 0) +
                   inducement +
                   entry_zone;

   features_array[index++] = (float)smc_score;
   return true;
  }
//+------------------------------------------------------------------+
