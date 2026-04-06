//+------------------------------------------------------------------+
//|                                          Features_Volatility.mqh |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property version   "1.0.0"

class CVolatilityFeatures
  {
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;
   int               m_hATR14;

   double            SampleStd(const double &values[], int count);

public:
                     CVolatilityFeatures() { m_hATR14 = INVALID_HANDLE; }
                    ~CVolatilityFeatures() {}

   bool              Init(string symbol, ENUM_TIMEFRAMES timeframe);
   void              Release();
   bool              Compute(int shift, int &index, float &features_array[], double atr14);
  };

bool CVolatilityFeatures::Init(string symbol, ENUM_TIMEFRAMES timeframe)
  {
   m_symbol = symbol;
   m_timeframe = timeframe;
   m_hATR14 = iATR(m_symbol, m_timeframe, 14);
   return (m_hATR14 != INVALID_HANDLE);
  }

void CVolatilityFeatures::Release()
  {
   if(m_hATR14 != INVALID_HANDLE)
     {
      IndicatorRelease(m_hATR14);
      m_hATR14 = INVALID_HANDLE;
     }
  }

double CVolatilityFeatures::SampleStd(const double &values[], int count)
  {
   if(count <= 1)
      return 0.0;

   double mean = 0.0;
   for(int i = 0; i < count; i++)
      mean += values[i];
   mean /= (double)count;

   double sum_sq = 0.0;
   for(int j = 0; j < count; j++)
      sum_sq += MathPow(values[j] - mean, 2.0);

   return MathSqrt(sum_sq / (double)(count - 1));
  }

bool CVolatilityFeatures::Compute(int shift, int &index, float &features_array[], double atr14)
  {
   double atr_values[240];
   int atr_below = 0;
   for(int i = 0; i < 240; i++)
     {
      double value[1];
      if(CopyBuffer(m_hATR14, 0, shift + i, 1, value) <= 0)
         return false;
      atr_values[i] = value[0];
      if(value[0] <= atr14)
         atr_below++;
     }
   double atr_percentile = (double)atr_below / 240.0;

   double close_values[10];
   for(int j = 0; j < 10; j++)
      close_values[j] = iClose(m_symbol, m_timeframe, shift + j);
   double tick_volatility = SampleStd(close_values, 10);

   double high_0 = iHigh(m_symbol, m_timeframe, shift);
   double low_0 = iLow(m_symbol, m_timeframe, shift);
   double high_1 = iHigh(m_symbol, m_timeframe, shift + 1);
   double low_1 = iLow(m_symbol, m_timeframe, shift + 1);
   double range_expansion = (high_0 - low_0) / ((high_1 - low_1) + 0.0001);

   double atr_std = SampleStd(atr_values, 240);
   double atr_mean = 0.0;
   for(int k = 0; k < 240; k++)
      atr_mean += atr_values[k];
   atr_mean /= 240.0;

   double true_range = high_0 - low_0;
   int tr_below = 0;
   for(int bar = 0; bar < 60; bar++)
     {
      double bar_true_range = iHigh(m_symbol, m_timeframe, shift + bar) - iLow(m_symbol, m_timeframe, shift + bar);
      if(bar_true_range <= true_range)
         tr_below++;
     }

   double close_0 = iClose(m_symbol, m_timeframe, shift);
   double close_3 = iClose(m_symbol, m_timeframe, shift + 3);
   double close_1 = iClose(m_symbol, m_timeframe, shift + 1);
   double close_4 = iClose(m_symbol, m_timeframe, shift + 4);
   double price_velocity = (close_0 - close_3) / 3.0;
   double prev_velocity = (close_1 - close_4) / 3.0;
   double price_acceleration = price_velocity - prev_velocity;

   features_array[index++] = (float)atr_percentile;
   features_array[index++] = (float)tick_volatility;
   features_array[index++] = (float)range_expansion;
   features_array[index++] = (float)((atr14 - atr_mean) / (atr_std + 0.0001));
   features_array[index++] = (float)true_range;
   features_array[index++] = (float)((double)tr_below / 60.0);
   features_array[index++] = (float)price_velocity;
   features_array[index++] = (float)price_acceleration;

   return true;
  }
//+------------------------------------------------------------------+
