//+------------------------------------------------------------------+
//|                                         Features_PriceAction.mqh |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property version   "1.0.0"

class CPriceActionFeatures
  {
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;

public:
                     CPriceActionFeatures() {}
                    ~CPriceActionFeatures() {}

   bool              Init(string symbol, ENUM_TIMEFRAMES timeframe)
     {
      m_symbol = symbol;
      m_timeframe = timeframe;
      return true;
     }
   bool              Compute(int shift, int &index, float &features_array[], double atr14);
  };

bool CPriceActionFeatures::Compute(int shift, int &index, float &features_array[], double atr14)
  {
   double close_0 = iClose(m_symbol, m_timeframe, shift);
   double close_1 = iClose(m_symbol, m_timeframe, shift + 1);
   double close_5 = iClose(m_symbol, m_timeframe, shift + 5);
   double close_15 = iClose(m_symbol, m_timeframe, shift + 15);
   double close_14 = iClose(m_symbol, m_timeframe, shift + 14);

   double high_50 = iHigh(m_symbol, m_timeframe, shift);
   double low_50 = iLow(m_symbol, m_timeframe, shift);
   for(int i = 1; i < 50; i++)
     {
      high_50 = MathMax(high_50, iHigh(m_symbol, m_timeframe, shift + i));
      low_50 = MathMin(low_50, iLow(m_symbol, m_timeframe, shift + i));
     }

   features_array[index++] = (float)((close_0 - close_1) / (close_1 + 0.0001));
   features_array[index++] = (float)((close_0 - close_5) / (close_5 + 0.0001));
   features_array[index++] = (float)((close_0 - close_15) / (close_15 + 0.0001));
   features_array[index++] = (float)(close_0 - close_14);
   features_array[index++] = (float)((high_50 - close_0) / (atr14 + 0.0001));
   features_array[index++] = (float)((close_0 - low_50) / (atr14 + 0.0001));
   features_array[index++] = 0.0f;

   return true;
  }
//+------------------------------------------------------------------+
