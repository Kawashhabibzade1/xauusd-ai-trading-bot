//+------------------------------------------------------------------+
//|                                           Features_Structure.mqh |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property version   "1.1.0"

class CStructureFeatures
  {
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;

public:
                     CStructureFeatures() {}
                    ~CStructureFeatures() {}

   bool              Init(string symbol, ENUM_TIMEFRAMES timeframe)
     {
      m_symbol = symbol;
      m_timeframe = timeframe;
      return true;
     }
   bool              Compute(int shift, int &index, float &features_array[]);
  };

bool CStructureFeatures::Compute(int shift, int &index, float &features_array[])
  {
   double close_0 = iClose(m_symbol, m_timeframe, shift);
   double open_0 = iOpen(m_symbol, m_timeframe, shift);
   double high_0 = iHigh(m_symbol, m_timeframe, shift);
   double low_0 = iLow(m_symbol, m_timeframe, shift);
   double close_1 = iClose(m_symbol, m_timeframe, shift + 1);
   double open_1 = iOpen(m_symbol, m_timeframe, shift + 1);
   double high_2 = iHigh(m_symbol, m_timeframe, shift + 2);
   double low_2 = iLow(m_symbol, m_timeframe, shift + 2);

   double rolling_high = high_0;
   double rolling_low = low_0;
   for(int i = 1; i < 5; i++)
     {
      rolling_high = MathMax(rolling_high, iHigh(m_symbol, m_timeframe, shift + i));
      rolling_low = MathMin(rolling_low, iLow(m_symbol, m_timeframe, shift + i));
     }

   int bullish_ob = (close_1 > open_1 && close_0 < open_0) ? 1 : 0;
   int bearish_ob = (close_1 < open_1 && close_0 > open_0) ? 1 : 0;
   double fvg_up = low_0 - high_2;
   double fvg_down = low_2 - high_0;
   double fvg_size = MathMax(fvg_up, fvg_down);

   double prev20_high = iHigh(m_symbol, m_timeframe, shift + 1);
   double prev20_low = iLow(m_symbol, m_timeframe, shift + 1);
   for(int j = 2; j < 21; j++)
     {
      prev20_high = MathMax(prev20_high, iHigh(m_symbol, m_timeframe, shift + j));
      prev20_low = MathMin(prev20_low, iLow(m_symbol, m_timeframe, shift + j));
     }

   int sweep_high = (high_0 > prev20_high && close_0 < open_0) ? 1 : 0;
   int sweep_low = (low_0 < prev20_low && close_0 > open_0) ? 1 : 0;

   double session_high = high_0;
   double session_low = low_0;
   for(int k = 1; k < 240; k++)
     {
      session_high = MathMax(session_high, iHigh(m_symbol, m_timeframe, shift + k));
      session_low = MathMin(session_low, iLow(m_symbol, m_timeframe, shift + k));
     }
   double premium_discount = (close_0 - session_low) / ((session_high - session_low) + 0.0001);

   features_array[index++] = (float)(rolling_high - close_0);
   features_array[index++] = (float)(close_0 - rolling_low);
   features_array[index++] = (float)bullish_ob;
   features_array[index++] = (float)bearish_ob;
   features_array[index++] = (float)fvg_up;
   features_array[index++] = (float)fvg_down;
   features_array[index++] = (float)fvg_size;
   features_array[index++] = (float)sweep_high;
   features_array[index++] = (float)sweep_low;
   features_array[index++] = (float)premium_discount;

   return true;
  }
//+------------------------------------------------------------------+
