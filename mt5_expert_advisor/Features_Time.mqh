//+------------------------------------------------------------------+
//|                                                Features_Time.mqh |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property version   "1.0.0"

class CTimeFeatures
  {
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;

public:
                     CTimeFeatures() {}
                    ~CTimeFeatures() {}

   bool              Init(string symbol, ENUM_TIMEFRAMES timeframe)
     {
      m_symbol = symbol;
      m_timeframe = timeframe;
      return true;
     }
   bool              Compute(int shift, int &index, float &features_array[]);
  };

bool CTimeFeatures::Compute(int shift, int &index, float &features_array[])
  {
   MqlDateTime dt;
   TimeToStruct(iTime(m_symbol, m_timeframe, shift), dt);

   int hour = dt.hour;
   int minute = dt.min;
   int dayofweek = dt.day_of_week;
   int minutes_since_midnight = hour * 60 + minute;
   int minutes_since_london = minutes_since_midnight - (8 * 60);
   int minutes_since_ny = minutes_since_midnight - (13 * 60 + 30);
   double session_position = (double)minutes_since_ny / (3.5 * 60.0);
   session_position = MathMax(0.0, MathMin(1.0, session_position));

   features_array[index++] = (float)hour;
   features_array[index++] = (float)minute;
   features_array[index++] = (float)dayofweek;
   features_array[index++] = (float)minutes_since_london;
   features_array[index++] = (float)minutes_since_ny;
   features_array[index++] = (float)session_position;

   return true;
  }
//+------------------------------------------------------------------+
