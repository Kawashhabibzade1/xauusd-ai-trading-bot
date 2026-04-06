//+------------------------------------------------------------------+
//|                                           Features_Orderflow.mqh |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property version   "1.1.0"

class COrderflowFeatures
  {
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;

public:
                     COrderflowFeatures() {}
                    ~COrderflowFeatures() {}

   bool              Init(string symbol, ENUM_TIMEFRAMES timeframe)
     {
      m_symbol = symbol;
      m_timeframe = timeframe;
      return true;
     }
   bool              Compute(int shift, int &index, float &features_array[], double atr14);
  };

bool COrderflowFeatures::Compute(int shift, int &index, float &features_array[], double atr14)
  {
   double close_0 = iClose(m_symbol, m_timeframe, shift);
   double open_0 = iOpen(m_symbol, m_timeframe, shift);
   double close_1 = iClose(m_symbol, m_timeframe, shift + 1);
   double bar_direction = (close_0 > open_0) ? 1.0 : (close_0 < open_0 ? -1.0 : 0.0);
   double delta = (double)iVolume(m_symbol, m_timeframe, shift) * bar_direction;

   int total_bars = Bars(m_symbol, m_timeframe);
   double cvd = 0.0;
   for(int bar = total_bars - 1; bar >= shift; bar--)
     {
      double close_bar = iClose(m_symbol, m_timeframe, bar);
      double open_bar = iOpen(m_symbol, m_timeframe, bar);
      double direction = (close_bar > open_bar) ? 1.0 : (close_bar < open_bar ? -1.0 : 0.0);
      cvd += (double)iVolume(m_symbol, m_timeframe, bar) * direction;
     }

   double cvd_prev = cvd - delta;
   double price_change = (close_1 != 0.0) ? ((close_0 - close_1) / close_1) : 0.0;
   double cvd_change = (cvd_prev != 0.0) ? ((cvd - cvd_prev) / cvd_prev) : 0.0;

   double volume_ma = 0.0;
   for(int i = 0; i < 20; i++)
      volume_ma += (double)iVolume(m_symbol, m_timeframe, shift + i);
   volume_ma /= 20.0;

   double volume_ratio = (double)iVolume(m_symbol, m_timeframe, shift) / (volume_ma + 0.0001);
   double price_range_norm = (iHigh(m_symbol, m_timeframe, shift) - iLow(m_symbol, m_timeframe, shift)) / (atr14 + 0.0001);
   double absorption_score = volume_ratio * (1.0 / (price_range_norm + 0.001));

   features_array[index++] = (float)bar_direction;
   features_array[index++] = (float)delta;
   features_array[index++] = (float)cvd;
   features_array[index++] = (float)price_change;
   features_array[index++] = (float)cvd_change;
   features_array[index++] = (float)(price_change - cvd_change);
   features_array[index++] = (float)volume_ma;
   features_array[index++] = (float)volume_ratio;
   features_array[index++] = (float)price_range_norm;
   features_array[index++] = (float)absorption_score;

   return true;
  }
//+------------------------------------------------------------------+
