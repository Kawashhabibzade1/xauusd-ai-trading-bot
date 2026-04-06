//+------------------------------------------------------------------+
//|                                                 Features_SMC.mqh |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property version   "1.0.0"

class CSMCFeatures
  {
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;

public:
                     CSMCFeatures() {}
                    ~CSMCFeatures() {}

   bool              Init(string symbol, ENUM_TIMEFRAMES timeframe)
     {
      m_symbol = symbol;
      m_timeframe = timeframe;
      return true;
     }
   bool              Compute(int shift, int &index, float &features_array[],
                             double sweep_high, double sweep_low,
                             double fvg_size, double bullish_ob, double bearish_ob);
  };

bool CSMCFeatures::Compute(int shift, int &index, float &features_array[],
                           double sweep_high, double sweep_low,
                           double fvg_size, double bullish_ob, double bearish_ob)
  {
   double close_0 = iClose(m_symbol, m_timeframe, shift);
   double close_240 = iClose(m_symbol, m_timeframe, shift + 240);
   double h4_high = iHigh(m_symbol, m_timeframe, shift);
   double h4_low = iLow(m_symbol, m_timeframe, shift);

   for(int i = 1; i < 240; i++)
     {
      h4_high = MathMax(h4_high, iHigh(m_symbol, m_timeframe, shift + i));
      h4_low = MathMin(h4_low, iLow(m_symbol, m_timeframe, shift + i));
     }

   double h4_mid = (h4_high + h4_low) / 2.0;
   int h4_bias = 0;
   if(close_0 > h4_mid && close_0 > close_240)
      h4_bias = 1;
   else if(close_0 < h4_mid && close_0 < close_240)
      h4_bias = -1;

   int in_discount = (close_0 < h4_mid) ? 1 : 0;
   int in_premium = (close_0 > h4_mid) ? 1 : 0;
   int inducement_taken = (sweep_high > 0.5 || sweep_low > 0.5) ? 1 : 0;
   int entry_zone_present = (fvg_size > 0.0 || bullish_ob > 0.5 || bearish_ob > 0.5) ? 1 : 0;
   int smc_quality_score =
      (h4_bias != 0 ? 1 : 0) +
      ((in_discount == 1 || in_premium == 1) ? 1 : 0) +
      inducement_taken +
      entry_zone_present;

   features_array[index++] = (float)h4_bias;
   features_array[index++] = (float)in_discount;
   features_array[index++] = (float)in_premium;
   features_array[index++] = (float)inducement_taken;
   features_array[index++] = (float)entry_zone_present;
   features_array[index++] = (float)smc_quality_score;

   return true;
  }
//+------------------------------------------------------------------+
