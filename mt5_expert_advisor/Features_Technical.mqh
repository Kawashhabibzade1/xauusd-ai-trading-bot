//+------------------------------------------------------------------+
//|                                           Features_Technical.mqh |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property version   "1.1.0"

class CTechFeatures
  {
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;
   int               m_hATR14;
   int               m_hATR5;
   int               m_hRSI14;
   int               m_hEMA12;
   int               m_hEMA26;
   int               m_hSMA50;
   int               m_hSMA200;
   int               m_hMACD;
   int               m_hBB;
   int               m_hStoch;

   double            GetBufferValue(int handle, int buffer_num, int shift);

public:
                     CTechFeatures() {}
                    ~CTechFeatures() {}

   bool              Init(string symbol, ENUM_TIMEFRAMES timeframe);
   void              Release();
   bool              Compute(int shift, int &index, float &features_array[]);
  };

bool CTechFeatures::Init(string symbol, ENUM_TIMEFRAMES timeframe)
  {
   m_symbol = symbol;
   m_timeframe = timeframe;

   m_hATR14 = iATR(m_symbol, m_timeframe, 14);
   m_hATR5 = iATR(m_symbol, m_timeframe, 5);
   m_hRSI14 = iRSI(m_symbol, m_timeframe, 14, PRICE_CLOSE);
   m_hEMA12 = iMA(m_symbol, m_timeframe, 12, 0, MODE_EMA, PRICE_CLOSE);
   m_hEMA26 = iMA(m_symbol, m_timeframe, 26, 0, MODE_EMA, PRICE_CLOSE);
   m_hSMA50 = iMA(m_symbol, m_timeframe, 50, 0, MODE_SMA, PRICE_CLOSE);
   m_hSMA200 = iMA(m_symbol, m_timeframe, 200, 0, MODE_SMA, PRICE_CLOSE);
   m_hMACD = iMACD(m_symbol, m_timeframe, 12, 26, 9, PRICE_CLOSE);
   m_hBB = iBands(m_symbol, m_timeframe, 20, 0, 2.0, PRICE_CLOSE);
   m_hStoch = iStochastic(m_symbol, m_timeframe, 14, 3, 3, MODE_SMA, STO_LOWHIGH);

   if(m_hATR14 == INVALID_HANDLE || m_hATR5 == INVALID_HANDLE || m_hRSI14 == INVALID_HANDLE)
      return false;
   if(m_hEMA12 == INVALID_HANDLE || m_hEMA26 == INVALID_HANDLE || m_hSMA50 == INVALID_HANDLE || m_hSMA200 == INVALID_HANDLE)
      return false;
   if(m_hMACD == INVALID_HANDLE || m_hBB == INVALID_HANDLE || m_hStoch == INVALID_HANDLE)
      return false;

   return true;
  }

void CTechFeatures::Release()
  {
   IndicatorRelease(m_hATR14);
   IndicatorRelease(m_hATR5);
   IndicatorRelease(m_hRSI14);
   IndicatorRelease(m_hEMA12);
   IndicatorRelease(m_hEMA26);
   IndicatorRelease(m_hSMA50);
   IndicatorRelease(m_hSMA200);
   IndicatorRelease(m_hMACD);
   IndicatorRelease(m_hBB);
   IndicatorRelease(m_hStoch);
  }

double CTechFeatures::GetBufferValue(int handle, int buffer_num, int shift)
  {
   double value[1];
   if(CopyBuffer(handle, buffer_num, shift, 1, value) <= 0)
      return 0.0;
   return value[0];
  }

bool CTechFeatures::Compute(int shift, int &index, float &features_array[])
  {
   double close_0 = iClose(m_symbol, m_timeframe, shift);
   double ema12_0 = GetBufferValue(m_hEMA12, 0, shift);
   double ema12_1 = GetBufferValue(m_hEMA12, 0, shift + 1);
   double ema26_0 = GetBufferValue(m_hEMA26, 0, shift);
   double ema26_1 = GetBufferValue(m_hEMA26, 0, shift + 1);
   double macd_main = GetBufferValue(m_hMACD, 0, shift);
   double macd_signal = GetBufferValue(m_hMACD, 1, shift);
   double bb_middle = GetBufferValue(m_hBB, 0, shift);
   double bb_upper = GetBufferValue(m_hBB, 1, shift);
   double bb_lower = GetBufferValue(m_hBB, 2, shift);
   double bb_width = bb_upper - bb_lower;
   double bb_position = (close_0 - bb_lower) / (bb_width + 0.0001);

   double vwap_num = 0.0;
   double vwap_den = 0.0;
   for(int i = 0; i < 60; i++)
     {
      double high_i = iHigh(m_symbol, m_timeframe, shift + i);
      double low_i = iLow(m_symbol, m_timeframe, shift + i);
      double close_i = iClose(m_symbol, m_timeframe, shift + i);
      double volume_i = (double)iVolume(m_symbol, m_timeframe, shift + i);
      double typical_price = (high_i + low_i + close_i) / 3.0;
      vwap_num += typical_price * volume_i;
      vwap_den += volume_i;
     }
   double vwap = vwap_num / (vwap_den + 0.0001);

   features_array[index++] = (float)GetBufferValue(m_hATR14, 0, shift);
   features_array[index++] = (float)GetBufferValue(m_hATR5, 0, shift);
   features_array[index++] = (float)GetBufferValue(m_hRSI14, 0, shift);
   features_array[index++] = (float)ema12_0;
   features_array[index++] = (float)ema26_0;
   features_array[index++] = (float)(ema12_0 - ema12_1);
   features_array[index++] = (float)(ema26_0 - ema26_1);
   features_array[index++] = (float)GetBufferValue(m_hSMA50, 0, shift);
   features_array[index++] = (float)GetBufferValue(m_hSMA200, 0, shift);
   features_array[index++] = (float)macd_main;
   features_array[index++] = (float)macd_signal;
   features_array[index++] = (float)(macd_main - macd_signal);
   features_array[index++] = (float)bb_upper;
   features_array[index++] = (float)bb_lower;
   features_array[index++] = (float)bb_middle;
   features_array[index++] = (float)bb_width;
   features_array[index++] = (float)bb_position;
   features_array[index++] = (float)vwap;
   features_array[index++] = (float)(close_0 / (vwap + 0.0001));
   features_array[index++] = (float)GetBufferValue(m_hStoch, 0, shift);
   features_array[index++] = (float)GetBufferValue(m_hStoch, 1, shift);

   return true;
  }
//+------------------------------------------------------------------+
