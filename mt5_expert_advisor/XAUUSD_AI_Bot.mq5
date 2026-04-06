//+------------------------------------------------------------------+
//|                                                XAUUSD_AI_Bot.mq5 |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property link      "https://github.com/andywarui/xauusd-ai-trading-bot"
#property version   "1.2.0"
#property description "Validation-first AI bot for XAUUSD using a LightGBM ONNX model"

#include "FeatureEngine.mqh"

input string   InpModelName            = "models\\xauusd_ai_v1.onnx";
input bool     InpValidationMode       = true;
input int      InpServerUtcOffsetHours = 0;
input bool     InpLogSignals           = true;
input double   InpConfidenceThresh     = 0.55;
input int      InpMagicNumber          = 202604;

long           ExtOnnxHandle = INVALID_HANDLE;
datetime       g_lastBarOpenTime = 0;
CFeatureEngine ExtFeatureEngine;

#define FEATURE_TOLERANCE 0.001
#define PROBABILITY_TOLERANCE 0.02
#define LOG_FILE "logs\\xauusd_ai_signals.csv"
#define VALIDATION_FILE "config\\validation_set.csv"

datetime ToUtc(datetime server_time)
  {
   return server_time - (InpServerUtcOffsetHours * 3600);
  }

bool IsOverlapBar(datetime utc_time)
  {
   MqlDateTime dt;
   TimeToStruct(utc_time, dt);
   if(dt.day_of_week == 0 || dt.day_of_week == 6)
      return false;
   return (dt.hour >= 13 && dt.hour <= 16);
  }

double FeatureChecksum(const float &features[])
  {
   double checksum = 0.0;
   for(int i = 0; i < FEATURE_COUNT; i++)
      checksum += (double)(i + 1) * (double)features[i];
   return checksum;
  }

bool IsDiscreteFeatureIndex(int index)
  {
   switch(index)
     {
      case 23:
      case 24:
      case 28:
      case 29:
      case 31:
      case 41:
      case 42:
      case 43:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
         return true;
      default:
         return false;
     }
  }

string ValidationStatusToString(int status)
  {
   switch(status)
     {
      case 1: return "matched";
      case 2: return "diff";
      case 3: return "missing";
      default: return "skipped";
     }
  }

void AppendSignalLog(datetime bar_time_utc, long predicted_class, const float &probabilities[], const float &features[], int validation_status)
  {
   if(!InpLogSignals)
      return;

   FolderCreate("logs");
   int handle = FileOpen(LOG_FILE, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_SHARE_READ);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open signal log file. Error: ", GetLastError());
      return;
     }

   if(FileSize(handle) == 0)
     {
      FileWrite(handle, "epoch_utc", "time_utc", "predicted_class", "prob_short", "prob_hold", "prob_long", "confidence", "feature_checksum", "validation_status");
     }

   FileSeek(handle, 0, SEEK_END);
   FileWrite(
      handle,
      (long)bar_time_utc,
      TimeToString(bar_time_utc, TIME_DATE | TIME_SECONDS),
      predicted_class,
      probabilities[0],
      probabilities[1],
      probabilities[2],
      MathMax(probabilities[0], MathMax(probabilities[1], probabilities[2])),
      FeatureChecksum(features),
      ValidationStatusToString(validation_status)
   );
   FileClose(handle);
  }

int CompareAgainstValidationFixture(datetime bar_time_utc, const float &features[], long predicted_class, const float &probabilities[])
  {
   int handle = FileOpen(VALIDATION_FILE, FILE_READ | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Validation fixture not available: ", VALIDATION_FILE);
      return 3;
     }

   bool skipped_header = false;
   while(!FileIsEnding(handle))
     {
      string epoch_field = FileReadString(handle);
      if(epoch_field == "")
        {
         if(FileIsEnding(handle))
            break;
        }

      string time_field = FileReadString(handle);
      double expected_features[FEATURE_COUNT];
      for(int i = 0; i < FEATURE_COUNT; i++)
         expected_features[i] = StringToDouble(FileReadString(handle));

      long expected_class = (long)StringToInteger(FileReadString(handle));
      double expected_short = StringToDouble(FileReadString(handle));
      double expected_hold = StringToDouble(FileReadString(handle));
      double expected_long = StringToDouble(FileReadString(handle));

      if(!skipped_header && epoch_field == "epoch_utc")
        {
         skipped_header = true;
         continue;
        }

      long expected_epoch = (long)StringToInteger(epoch_field);
      if(expected_epoch != (long)bar_time_utc)
         continue;

      bool feature_match = true;
      for(int idx = 0; idx < FEATURE_COUNT; idx++)
        {
         double actual = (double)features[idx];
         double expected = expected_features[idx];
         if(IsDiscreteFeatureIndex(idx))
           {
            if((long)MathRound(actual) != (long)MathRound(expected))
              {
               PrintFormat("Discrete feature mismatch @%d (%s): actual=%.6f expected=%.6f", idx, time_field, actual, expected);
               feature_match = false;
              }
           }
         else
           {
            if(MathAbs(actual - expected) > FEATURE_TOLERANCE)
              {
               PrintFormat("Feature mismatch @%d (%s): actual=%.6f expected=%.6f", idx, time_field, actual, expected);
               feature_match = false;
              }
           }
        }

      bool class_match = (predicted_class == expected_class);
      bool probability_match =
         (MathAbs((double)probabilities[0] - expected_short) <= PROBABILITY_TOLERANCE) &&
         (MathAbs((double)probabilities[1] - expected_hold) <= PROBABILITY_TOLERANCE) &&
         (MathAbs((double)probabilities[2] - expected_long) <= PROBABILITY_TOLERANCE);

      FileClose(handle);
      if(feature_match && class_match && probability_match)
         return 1;

      PrintFormat("Validation mismatch for %s: class=%d/%d prob=[%.4f,%.4f,%.4f]", time_field, predicted_class, expected_class, probabilities[0], probabilities[1], probabilities[2]);
      return 2;
     }

   FileClose(handle);
   Print("No matching validation fixture row for epoch ", (long)bar_time_utc);
   return 3;
  }

int OnInit()
  {
   Print("--- XAUUSD AI Bot v", __property_version, " Initializing ---");

   if(!InpValidationMode && !TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
     {
      Print("Auto-trading is disabled and validation mode is off.");
      return INIT_FAILED;
     }

   ExtOnnxHandle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   if(ExtOnnxHandle == INVALID_HANDLE)
     {
      PrintFormat("Failed to load ONNX model '%s'. Error=%d", InpModelName, GetLastError());
      return INIT_FAILED;
     }

   long input_shape[] = {1, FEATURE_COUNT};
   if(!OnnxSetInputShape(ExtOnnxHandle, 0, input_shape))
      Print("Warning: OnnxSetInputShape failed. Error: ", GetLastError());

   if(!ExtFeatureEngine.Init(_Symbol, PERIOD_CURRENT))
     {
      Print("Feature engine initialization failed.");
      return INIT_FAILED;
     }

   FolderCreate("logs");
   Print("Validation mode: ", (InpValidationMode ? "ON" : "OFF"));
   Print("Model loaded: ", InpModelName);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   ExtFeatureEngine.Release();
   if(ExtOnnxHandle != INVALID_HANDLE)
     {
      OnnxRelease(ExtOnnxHandle);
      ExtOnnxHandle = INVALID_HANDLE;
     }
  }

void OnTick()
  {
   datetime current_bar_open = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_bar_open == g_lastBarOpenTime)
      return;
   g_lastBarOpenTime = current_bar_open;

   datetime closed_bar_server_time = iTime(_Symbol, PERIOD_CURRENT, 1);
   datetime closed_bar_utc = ToUtc(closed_bar_server_time);
   if(!IsOverlapBar(closed_bar_utc))
      return;

   float features[FEATURE_COUNT];
   if(!ExtFeatureEngine.ComputeFeatures(1, features))
     {
      Print("Feature computation failed.");
      return;
     }

   long prediction_class[1];
   float probabilities[3];
   if(!OnnxRun(ExtOnnxHandle, ONNX_NO_CONVERSION, features, prediction_class, probabilities))
     {
      Print("ONNX inference failed. Error: ", GetLastError());
      return;
     }

   int validation_status = 0;
   if(InpValidationMode)
      validation_status = CompareAgainstValidationFixture(closed_bar_utc, features, prediction_class[0], probabilities);

   AppendSignalLog(closed_bar_utc, prediction_class[0], probabilities, features, validation_status);

   PrintFormat(
      "UTC=%s class=%d probs=[%.4f, %.4f, %.4f] confidence=%.4f validation=%s",
      TimeToString(closed_bar_utc, TIME_DATE | TIME_SECONDS),
      prediction_class[0],
      probabilities[0],
      probabilities[1],
      probabilities[2],
      MathMax(probabilities[0], MathMax(probabilities[1], probabilities[2])),
      ValidationStatusToString(validation_status)
   );

   if(!InpValidationMode)
     {
      Print("Trading mode is not implemented in this recovery pass.");
     }
  }
//+------------------------------------------------------------------+
