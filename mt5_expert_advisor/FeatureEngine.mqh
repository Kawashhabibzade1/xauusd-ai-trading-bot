//+------------------------------------------------------------------+
//|                                                FeatureEngine.mqh |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property version   "1.2.0"

#include "Features_Technical.mqh"
#include "Features_Structure.mqh"
#include "Features_Orderflow.mqh"
#include "Features_Time.mqh"
#include "Features_Volatility.mqh"
#include "Features_PriceAction.mqh"
#include "Features_SMC.mqh"

#define FEATURE_COUNT 68

class CFeatureEngine
  {
private:
   CTechFeatures         m_tech;
   CStructureFeatures    m_structure;
   COrderflowFeatures    m_orderflow;
   CTimeFeatures         m_time;
   CVolatilityFeatures   m_volatility;
   CPriceActionFeatures  m_price_action;
   CSMCFeatures          m_smc;

public:
                     CFeatureEngine() {}
                    ~CFeatureEngine() { Release(); }

   bool              Init(string symbol, ENUM_TIMEFRAMES timeframe);
   void              Release();
   bool              ComputeFeatures(int shift, float &features_array[]);
  };

bool CFeatureEngine::Init(string symbol, ENUM_TIMEFRAMES timeframe)
  {
   if(!m_tech.Init(symbol, timeframe))       return false;
   if(!m_structure.Init(symbol, timeframe))  return false;
   if(!m_orderflow.Init(symbol, timeframe))  return false;
   if(!m_time.Init(symbol, timeframe))       return false;
   if(!m_volatility.Init(symbol, timeframe)) return false;
   if(!m_price_action.Init(symbol, timeframe)) return false;
   if(!m_smc.Init(symbol, timeframe))        return false;
   return true;
  }

void CFeatureEngine::Release()
  {
   m_tech.Release();
   m_volatility.Release();
  }

bool CFeatureEngine::ComputeFeatures(int shift, float &features_array[])
  {
   if(ArraySize(features_array) < FEATURE_COUNT)
     {
      PrintFormat("Feature array too small: %d < %d", ArraySize(features_array), FEATURE_COUNT);
      return false;
     }

   ArrayInitialize(features_array, 0.0f);
   int index = 0;

   if(!m_tech.Compute(shift, index, features_array))
      return false;

   if(!m_structure.Compute(shift, index, features_array))
      return false;

   double atr14 = (double)features_array[0];
   double bullish_ob = (double)features_array[23];
   double bearish_ob = (double)features_array[24];
   double fvg_size = (double)features_array[27];
   double sweep_high = (double)features_array[28];
   double sweep_low = (double)features_array[29];

   if(!m_orderflow.Compute(shift, index, features_array, atr14))
      return false;

   if(!m_time.Compute(shift, index, features_array))
      return false;

   if(!m_volatility.Compute(shift, index, features_array, atr14))
      return false;

   if(!m_price_action.Compute(shift, index, features_array, atr14))
      return false;

   if(!m_smc.Compute(shift, index, features_array, sweep_high, sweep_low, fvg_size, bullish_ob, bearish_ob))
      return false;

   if(index != FEATURE_COUNT)
     {
      PrintFormat("Feature count mismatch: expected %d, computed %d", FEATURE_COUNT, index);
      return false;
     }

   return true;
  }
//+------------------------------------------------------------------+
