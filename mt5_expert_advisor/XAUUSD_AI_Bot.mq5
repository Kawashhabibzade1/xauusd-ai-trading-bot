//+------------------------------------------------------------------+
//|                                                XAUUSD_AI_Bot.mq5 |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property link      "https://github.com/andywarui/xauusd-ai-trading-bot"
#property version   "1.304"
#property description "Validation-first AI bot for XAUUSD using a LightGBM ONNX model with demo-only execution mode"

#include <Trade/Trade.mqh>
#include "FeatureEngine.mqh"

input string   InpModelName            = "models\\xauusd_ai_mt5_live.onnx";
input bool     InpValidationMode       = false;
input int      InpServerUtcOffsetHours = 0;
input bool     InpLogSignals           = true;
input double   InpConfidenceThresh     = 0.55;
input int      InpMagicNumber          = 202604;
input bool     InpEnableDemoTrading    = true;
input bool     InpDemoOnly             = true;
input bool     InpRequireTradeDirective = true;
input int      InpMaxDirectiveEntryDriftPoints = 30;
input int      InpDirectiveMaxAgeBars   = 2;
input int      InpSessionTradeLimit     = 0;
input bool     InpUseRiskBasedSizing   = true;
input double   InpRiskPerTradePercent  = 0.25;
input double   InpDemoMaxLotSize       = 0.10;
input double   InpFixedLotSize         = 0.01;
input double   InpStopAtrMultiple      = 1.00;
input double   InpTakeProfitRR         = 1.50;
input double   InpTp1ProfitLockR       = 0.25;
input int      InpMaxSpreadPoints      = 80;
input int      InpSlippagePoints       = 30;
input int      InpStopBufferPoints     = 10;
input bool     InpAllowSignalFlip      = false;
input string   InpTradeComment         = "XAUUSD_AI_Demo";

long           ExtOnnxHandle = INVALID_HANDLE;
datetime       g_lastBarOpenTime = 0;
datetime       g_pendingTradeBarUtc = 0;
long           g_pendingPredictedClass = 1;
float          g_pendingProbabilities[3];
bool           g_hasPendingTrade = false;
int            g_successfulTradeCount = 0;
int            g_hATR14 = INVALID_HANDLE;
datetime       g_lastAwaitingBarUtc = 0;
string         g_lastAwaitingMessage = "";
datetime       g_lastConsumedDirectiveUtc = 0;
double         g_activeTradeTp1 = 0.0;
double         g_activeTradeEntry = 0.0;
bool           g_activeTradeBreakevenDone = false;
int            g_activeTradeDirection = 0;
double         g_activeTradeRiskDistance = 0.0;
datetime       g_lastBreakevenDebugUtc = 0;
string         g_lastBreakevenDebugNote = "";
CTrade         ExtTrade;
CFeatureEngine ExtFeatureEngine;

#define FEATURE_TOLERANCE 0.001
#define PROBABILITY_TOLERANCE 0.02
#define LOG_FILE "logs\\xauusd_ai_signals.csv"
#define TRADE_LOG_FILE "logs\\xauusd_ai_trades.csv"
#define VALIDATION_FILE "config\\validation_set.csv"
#define TRADE_DIRECTIVE_FILE "config\\mt5_trade_directive.csv"
#define ACCOUNT_SNAPSHOT_FILE "config\\mt5_account_snapshot.csv"
#define LIVE_MODEL_FILE "models\\xauusd_ai_mt5_live.onnx"
#define BOT_VERSION "1.304"

struct TradeDirective
  {
   datetime time_utc;
   string recommended_trade;
   string gate_status;
   string paper_status;
   string paper_reason_blocked;
   double directional_confidence;
   double setup_score;
   double expected_value;
   double entry_price;
   double stop_loss;
   double tp1;
   double tp2;
   double volume_lots;
   double risk_cash;
   string position_sizing_mode;
   double account_balance_snapshot;
   double assumed_contract_size;
  };

string ResolveModelName()
  {
   if(InpEnableDemoTrading && InpModelName == "models\\xauusd_ai_v1.onnx" && FileIsExist(LIVE_MODEL_FILE))
      return LIVE_MODEL_FILE;
   return InpModelName;
  }

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

string PredictionLabel(long predicted_class)
  {
   switch((int)predicted_class)
     {
      case 0: return "SHORT";
      case 1: return "HOLD";
      case 2: return "LONG";
      default: return "UNKNOWN";
     }
  }

int PredictionDirection(long predicted_class)
  {
   if(predicted_class == 2)
      return 1;
   if(predicted_class == 0)
      return -1;
   return 0;
  }

long PredictedClassFromDirection(int direction)
  {
   if(direction > 0)
      return 2;
   if(direction < 0)
      return 0;
   return 1;
  }

string DirectionToString(int direction)
  {
   if(direction > 0)
      return "LONG";
   if(direction < 0)
      return "SHORT";
   return "HOLD";
  }

string PositionTypeToString(long position_type)
  {
   if(position_type == POSITION_TYPE_BUY)
      return "LONG";
   if(position_type == POSITION_TYPE_SELL)
      return "SHORT";
   return "UNKNOWN";
  }

bool IsDemoAccount()
  {
   return ((ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_TRADE_MODE_DEMO);
  }

void WriteAccountSnapshot()
  {
   FolderCreate("config");
   int handle = FileOpen(ACCOUNT_SNAPSHOT_FILE, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
      return;

   FileWrite(
      handle,
      "time_utc",
      "symbol",
      "login",
      "server",
      "currency",
      "balance",
      "equity",
      "leverage",
      "contract_size",
      "volume_min",
      "volume_max",
      "volume_step"
   );
   FileWrite(
      handle,
      TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS),
      _Symbol,
      (string)AccountInfoInteger(ACCOUNT_LOGIN),
      AccountInfoString(ACCOUNT_SERVER),
      AccountInfoString(ACCOUNT_CURRENCY),
      DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2),
      DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2),
      (string)AccountInfoInteger(ACCOUNT_LEVERAGE),
      DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE), 2),
      DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN), 2),
      DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX), 2),
      DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP), 2)
   );
   FileClose(handle);
  }

void LogAwaitingCondition(datetime bar_time_utc, string message)
  {
   if(g_lastAwaitingBarUtc == bar_time_utc && g_lastAwaitingMessage == message)
      return;
   g_lastAwaitingBarUtc = bar_time_utc;
   g_lastAwaitingMessage = message;
   Print(message);
  }

int VolumeDigits(double step)
  {
   int digits = 0;
   double value = step;
   while(digits < 8 && MathAbs(value - MathRound(value)) > 1e-8)
     {
      value *= 10.0;
      digits++;
     }
   return digits;
  }

double NormalizeVolume(double requested_volume)
  {
   double volume_min = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double volume_max = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(volume_min <= 0.0 || volume_step <= 0.0)
      return 0.0;

   double clipped = MathMax(volume_min, MathMin(volume_max, requested_volume));
   double stepped = volume_min + MathFloor(((clipped - volume_min) / volume_step) + 0.5) * volume_step;
   stepped = MathMax(volume_min, MathMin(volume_max, stepped));
   return NormalizeDouble(stepped, VolumeDigits(volume_step));
  }

double NormalizePrice(double price)
  {
   return NormalizeDouble(price, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
  }

bool PricesMatch(double left, double right, double tolerance_points = 2.0)
  {
   return (MathAbs(left - right) <= (_Point * tolerance_points));
  }

double ResolveRiskFraction()
  {
   double risk_fraction = InpRiskPerTradePercent / 100.0;
   return MathMax(0.0, risk_fraction);
  }

double CalculateRiskBasedVolume(int direction, double entry_price, double stop_loss, string &reason)
  {
   if(direction == 0)
     {
      reason = "hold_signal";
      return 0.0;
     }

   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(account_balance <= 0.0)
     {
      reason = "invalid_account_balance";
      return 0.0;
     }

   double risk_fraction = ResolveRiskFraction();
   if(risk_fraction <= 0.0)
     {
      reason = "invalid_risk_fraction";
      return 0.0;
     }

   ENUM_ORDER_TYPE order_type = (direction > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   double stop_loss_value_for_one_lot = 0.0;
   if(!OrderCalcProfit(order_type, _Symbol, 1.0, entry_price, stop_loss, stop_loss_value_for_one_lot))
     {
      reason = "order_calc_profit_failed";
      return 0.0;
     }

   double loss_per_lot = MathAbs(stop_loss_value_for_one_lot);
   if(loss_per_lot <= 0.0)
     {
      reason = "invalid_loss_per_lot";
      return 0.0;
     }

   double risk_cash = account_balance * risk_fraction;
   if(risk_cash <= 0.0)
     {
      reason = "invalid_risk_cash";
      return 0.0;
     }

   return (risk_cash / loss_per_lot);
  }

bool CopyAtr14(double &atr_value)
  {
   if(g_hATR14 == INVALID_HANDLE)
      return false;

   double atr_buffer[1];
   if(CopyBuffer(g_hATR14, 0, 1, 1, atr_buffer) <= 0)
      return false;

   atr_value = atr_buffer[0];
   return (atr_value > 0.0);
  }

double CurrentSpreadPoints(const MqlTick &tick)
  {
   return (tick.ask - tick.bid) / _Point;
  }

bool IsTradeModeAllowed(int direction)
  {
   ENUM_SYMBOL_TRADE_MODE trade_mode = (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE);
   if(trade_mode == SYMBOL_TRADE_MODE_DISABLED || trade_mode == SYMBOL_TRADE_MODE_CLOSEONLY)
      return false;
   if(direction > 0 && trade_mode == SYMBOL_TRADE_MODE_SHORTONLY)
      return false;
   if(direction < 0 && trade_mode == SYMBOL_TRADE_MODE_LONGONLY)
      return false;
   return true;
  }

int CountManagedPositions(ulong &ticket, long &position_type)
  {
   int count = 0;
   ticket = 0;
   position_type = -1;

   for(int index = PositionsTotal() - 1; index >= 0; index--)
     {
      ulong current_ticket = PositionGetTicket(index);
      if(current_ticket == 0)
         continue;
      if(!PositionSelectByTicket(current_ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      count++;
      ticket = current_ticket;
      position_type = PositionGetInteger(POSITION_TYPE);
     }

   return count;
  }

bool EnsureManagedPositionProtection(
   datetime bar_time_utc,
   int expected_direction,
   double expected_stop_loss,
   double expected_take_profit
)
  {
   const int max_attempts = 10;
   for(int attempt = 0; attempt < max_attempts; attempt++)
     {
      ulong ticket = 0;
      long position_type = -1;
      int managed_positions = CountManagedPositions(ticket, position_type);
      if(managed_positions == 1 && PositionSelectByTicket(ticket))
        {
         int actual_direction = 0;
         if(position_type == POSITION_TYPE_BUY)
            actual_direction = 1;
         else if(position_type == POSITION_TYPE_SELL)
            actual_direction = -1;

         if(expected_direction != 0 && actual_direction != expected_direction)
           {
            Sleep(200);
            continue;
           }

         double current_stop_loss = PositionGetDouble(POSITION_SL);
         double current_take_profit = PositionGetDouble(POSITION_TP);
         bool stop_matches = (current_stop_loss > 0.0 && PricesMatch(current_stop_loss, expected_stop_loss));
         bool take_matches = (current_take_profit > 0.0 && PricesMatch(current_take_profit, expected_take_profit));
         if(stop_matches && take_matches)
            return true;

         if(!ExtTrade.PositionModify(ticket, expected_stop_loss, expected_take_profit))
           {
            AppendTradeLog(
               bar_time_utc,
               "PROTECT_REJECT",
               DirectionToString(actual_direction),
               0.0,
               PositionGetDouble(POSITION_VOLUME),
               PositionGetDouble(POSITION_PRICE_OPEN),
               expected_stop_loss,
               expected_take_profit,
               0.0,
               ExtTrade.ResultRetcode(),
               ExtTrade.ResultRetcodeDescription(),
               "post_open_protection_failed"
            );
            Sleep(200);
            continue;
           }

         if(PositionSelectByTicket(ticket))
           {
            current_stop_loss = PositionGetDouble(POSITION_SL);
            current_take_profit = PositionGetDouble(POSITION_TP);
            stop_matches = (current_stop_loss > 0.0 && PricesMatch(current_stop_loss, expected_stop_loss));
            take_matches = (current_take_profit > 0.0 && PricesMatch(current_take_profit, expected_take_profit));
            if(stop_matches && take_matches)
              {
               AppendTradeLog(
                  bar_time_utc,
                  "PROTECT",
                  DirectionToString(actual_direction),
                  0.0,
                  PositionGetDouble(POSITION_VOLUME),
                  PositionGetDouble(POSITION_PRICE_OPEN),
                  current_stop_loss,
                  current_take_profit,
                  0.0,
                  ExtTrade.ResultRetcode(),
                  ExtTrade.ResultRetcodeDescription(),
                  "post_open_protection_applied"
               );
               return true;
              }
           }
        }

      Sleep(200);
     }

   AppendTradeLog(
      bar_time_utc,
      "PROTECT_REJECT",
      DirectionToString(expected_direction),
      0.0,
      0.0,
      0.0,
      expected_stop_loss,
      expected_take_profit,
      0.0,
      0,
      "",
      "post_open_position_not_confirmed"
   );
   return false;
  }

void ResetManagedTradeState()
  {
   g_activeTradeTp1 = 0.0;
   g_activeTradeEntry = 0.0;
   g_activeTradeBreakevenDone = false;
   g_activeTradeDirection = 0;
   g_activeTradeRiskDistance = 0.0;
   g_lastBreakevenDebugUtc = 0;
   g_lastBreakevenDebugNote = "";
  }

void MaybeLogBreakevenDebug(
   long position_type,
   double volume,
   double current_stop_loss,
   double current_take_profit,
   double protected_stop,
   const MqlTick &tick,
   bool tp1_reached,
   string stage
)
  {
   datetime now_utc = ToUtc(TimeCurrent());
   string note = StringFormat(
      "%s|entry=%.2f|tp1=%.2f|sl=%.2f|target_sl=%.2f|tp=%.2f|bid=%.2f|ask=%.2f|risk=%.2f|tp1_reached=%s|done=%s",
      stage,
      g_activeTradeEntry,
      g_activeTradeTp1,
      current_stop_loss,
      protected_stop,
      current_take_profit,
      tick.bid,
      tick.ask,
      g_activeTradeRiskDistance,
      (tp1_reached ? "true" : "false"),
      (g_activeTradeBreakevenDone ? "true" : "false")
   );
   if(g_lastBreakevenDebugUtc == now_utc && g_lastBreakevenDebugNote == note)
      return;
   g_lastBreakevenDebugUtc = now_utc;
   g_lastBreakevenDebugNote = note;
   AppendTradeLog(
      now_utc,
      "BREAKEVEN_DEBUG",
      PositionTypeToString(position_type),
      0.0,
      volume,
      g_activeTradeEntry,
      current_stop_loss,
      current_take_profit,
      0.0,
      0,
      "",
      note
   );
  }

void SeedManagedTradeStateFromPosition()
  {
   ulong ticket = 0;
   long position_type = -1;
   int managed_positions = CountManagedPositions(ticket, position_type);
   if(managed_positions != 1 || !PositionSelectByTicket(ticket))
     {
      ResetManagedTradeState();
      return;
     }

   g_activeTradeEntry = PositionGetDouble(POSITION_PRICE_OPEN);
   g_activeTradeDirection = (position_type == POSITION_TYPE_BUY ? 1 : (position_type == POSITION_TYPE_SELL ? -1 : 0));

   double current_stop = PositionGetDouble(POSITION_SL);
   g_activeTradeRiskDistance = MathAbs(g_activeTradeEntry - current_stop);
   g_activeTradeBreakevenDone = false;

   if(g_activeTradeTp1 > 0.0 && g_activeTradeDirection != 0)
      return;

   TradeDirective directive;
   if(!LoadLatestTradeDirective(directive))
     {
      double current_take_profit = PositionGetDouble(POSITION_TP);
      if(current_take_profit > 0.0)
         g_activeTradeTp1 = current_take_profit;
      return;
     }

   int directive_direction = DirectiveDirection(directive);
   if(directive_direction != g_activeTradeDirection || directive_direction == 0)
     {
      double current_take_profit = PositionGetDouble(POSITION_TP);
      if(current_take_profit > 0.0)
         g_activeTradeTp1 = current_take_profit;
      return;
     }

   double current_take_profit = PositionGetDouble(POSITION_TP);
   double base_tp1_distance = MathAbs(directive.tp1 - directive.entry_price);
   double base_tp2_reference = (directive.tp2 > 0.0 ? directive.tp2 : directive.tp1);
   double base_tp2_distance = MathAbs(base_tp2_reference - directive.entry_price);
   double mapped_tp1_distance = base_tp1_distance;
   if(current_take_profit > 0.0 && base_tp1_distance > 0.0 && base_tp2_distance > 0.0)
     {
      double current_tp_distance = MathAbs(current_take_profit - g_activeTradeEntry);
      mapped_tp1_distance = current_tp_distance * (base_tp1_distance / base_tp2_distance);
     }

   if(g_activeTradeDirection > 0)
      g_activeTradeTp1 = g_activeTradeEntry + mapped_tp1_distance;
   else
      g_activeTradeTp1 = g_activeTradeEntry - mapped_tp1_distance;
  }

void ManageOpenPositionBreakeven()
  {
   ulong ticket = 0;
   long position_type = -1;
   int managed_positions = CountManagedPositions(ticket, position_type);
   if(managed_positions != 1)
     {
      ResetManagedTradeState();
      return;
     }
   if(!PositionSelectByTicket(ticket))
     {
      ResetManagedTradeState();
      return;
     }

   if(g_activeTradeEntry <= 0.0 || g_activeTradeDirection == 0)
      SeedManagedTradeStateFromPosition();

   if(g_activeTradeBreakevenDone || g_activeTradeTp1 <= 0.0 || g_activeTradeEntry <= 0.0 || g_activeTradeDirection == 0)
      return;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
      return;

   bool tp1_reached = false;
   if(g_activeTradeDirection > 0)
      tp1_reached = (tick.bid >= g_activeTradeTp1);
   else
      tp1_reached = (tick.ask <= g_activeTradeTp1);

   double current_take_profit = PositionGetDouble(POSITION_TP);
   double current_stop_loss = PositionGetDouble(POSITION_SL);
   if(g_activeTradeRiskDistance <= 0.0)
      g_activeTradeRiskDistance = MathAbs(g_activeTradeEntry - current_stop_loss);

   double protected_stop = g_activeTradeEntry;
   double profit_lock_distance = g_activeTradeRiskDistance * MathMax(InpTp1ProfitLockR, 0.0);
   if(g_activeTradeDirection > 0)
      protected_stop = NormalizePrice(g_activeTradeEntry + profit_lock_distance);
   else
      protected_stop = NormalizePrice(g_activeTradeEntry - profit_lock_distance);

   MaybeLogBreakevenDebug(
      position_type,
      PositionGetDouble(POSITION_VOLUME),
      current_stop_loss,
      current_take_profit,
      protected_stop,
      tick,
      tp1_reached,
      "evaluate"
   );

   if(!tp1_reached)
      return;

   double tolerance = _Point * 2.0;
   if((g_activeTradeDirection > 0 && current_stop_loss >= (protected_stop - tolerance)) ||
      (g_activeTradeDirection < 0 && current_stop_loss <= (protected_stop + tolerance)))
     {
      g_activeTradeBreakevenDone = true;
      return;
     }

   if(!ExtTrade.PositionModify(ticket, protected_stop, current_take_profit))
     {
      AppendTradeLog(
         ToUtc(TimeCurrent()),
         "BREAKEVEN_REJECT",
         PositionTypeToString(position_type),
         0.0,
         PositionGetDouble(POSITION_VOLUME),
         g_activeTradeEntry,
         protected_stop,
         current_take_profit,
         0.0,
         ExtTrade.ResultRetcode(),
         ExtTrade.ResultRetcodeDescription(),
         "tp1_profit_lock_failed"
      );
      return;
     }

   g_activeTradeBreakevenDone = true;
   AppendTradeLog(
      ToUtc(TimeCurrent()),
      "BREAKEVEN",
      PositionTypeToString(position_type),
      0.0,
      PositionGetDouble(POSITION_VOLUME),
      g_activeTradeEntry,
      protected_stop,
      current_take_profit,
      0.0,
      ExtTrade.ResultRetcode(),
      ExtTrade.ResultRetcodeDescription(),
      "tp1_profit_lock"
   );
  }

int DirectiveDirection(const TradeDirective &directive)
  {
   if(directive.recommended_trade == "LONG")
      return 1;
   if(directive.recommended_trade == "SHORT")
      return -1;
   return 0;
  }

bool LoadLatestTradeDirective(TradeDirective &directive)
  {
   directive.time_utc = 0;
   directive.recommended_trade = "";
   directive.gate_status = "";
   directive.paper_status = "";
   directive.paper_reason_blocked = "";
   directive.directional_confidence = 0.0;
   directive.setup_score = 0.0;
   directive.expected_value = 0.0;
   directive.entry_price = 0.0;
   directive.stop_loss = 0.0;
   directive.tp1 = 0.0;
   directive.tp2 = 0.0;
   directive.volume_lots = 0.0;
   directive.risk_cash = 0.0;
   directive.position_sizing_mode = "";
   directive.account_balance_snapshot = 0.0;
   directive.assumed_contract_size = 0.0;

   int handle = FileOpen(TRADE_DIRECTIVE_FILE, FILE_READ | FILE_CSV | FILE_ANSI | FILE_SHARE_READ | FILE_SHARE_WRITE, ',');
   if(handle == INVALID_HANDLE)
      return false;

   bool skipped_header = false;
   bool found = false;
   while(!FileIsEnding(handle))
     {
      string epoch_field = FileReadString(handle);
      if(epoch_field == "")
        {
         if(FileIsEnding(handle))
            break;
        }

      string time_field = FileReadString(handle);
      string recommended_trade_field = FileReadString(handle);
      string gate_status_field = FileReadString(handle);
      string paper_status_field = FileReadString(handle);
      string paper_reason_blocked_field = FileReadString(handle);
      string directional_confidence_field = FileReadString(handle);
      string setup_score_field = FileReadString(handle);
      string expected_value_field = FileReadString(handle);
      string entry_price_field = FileReadString(handle);
      string stop_loss_field = FileReadString(handle);
      string tp1_field = FileReadString(handle);
      string tp2_field = FileReadString(handle);
      FileReadString(handle); // session_name
      FileReadString(handle); // manual_override_used
      FileReadString(handle); // execution_scope
      FileReadString(handle); // learning_source
      FileReadString(handle); // streamlit_scope
      if(!FileIsLineEnding(handle))
         directive.volume_lots = StringToDouble(FileReadString(handle));
      if(!FileIsLineEnding(handle))
         directive.risk_cash = StringToDouble(FileReadString(handle));
      if(!FileIsLineEnding(handle))
         directive.position_sizing_mode = FileReadString(handle);
      if(!FileIsLineEnding(handle))
         directive.account_balance_snapshot = StringToDouble(FileReadString(handle));
      if(!FileIsLineEnding(handle))
         directive.assumed_contract_size = StringToDouble(FileReadString(handle));

      if(!skipped_header && epoch_field == "epoch_utc")
        {
         skipped_header = true;
         continue;
        }

      long epoch_value = StringToInteger(epoch_field);
      if(epoch_value <= 0 && time_field != "")
         epoch_value = (long)StringToTime(time_field);
      if(epoch_value <= 0 || recommended_trade_field == "" || gate_status_field == "" || paper_status_field == "")
         continue;

      directive.time_utc = (datetime)epoch_value;
      directive.recommended_trade = recommended_trade_field;
      directive.gate_status = gate_status_field;
      directive.paper_status = paper_status_field;
      directive.paper_reason_blocked = paper_reason_blocked_field;
      directive.directional_confidence = StringToDouble(directional_confidence_field);
      directive.setup_score = StringToDouble(setup_score_field);
      directive.expected_value = StringToDouble(expected_value_field);
      directive.entry_price = StringToDouble(entry_price_field);
      directive.stop_loss = StringToDouble(stop_loss_field);
      directive.tp1 = StringToDouble(tp1_field);
      directive.tp2 = StringToDouble(tp2_field);
      found = true;
     }

   FileClose(handle);
   return found;
  }

void AppendTradeLog(
   datetime bar_time_utc,
   string action,
   string direction,
   double confidence,
   double volume,
   double entry_price,
   double stop_loss,
   double take_profit,
   double spread_points,
   ulong retcode,
   string retcode_desc,
   string note
)
  {
   FolderCreate("logs");
   int handle = FileOpen(TRADE_LOG_FILE, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_SHARE_READ, ',');
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open trade log file. Error: ", GetLastError());
      return;
     }

   if(FileSize(handle) == 0)
     {
      FileWrite(handle, "epoch_utc", "time_utc", "action", "direction", "confidence", "volume", "entry_price", "stop_loss", "take_profit", "spread_points", "retcode", "retcode_desc", "note");
     }

   FileSeek(handle, 0, SEEK_END);
   FileWrite(
      handle,
      (long)bar_time_utc,
      TimeToString(bar_time_utc, TIME_DATE | TIME_SECONDS),
      action,
      direction,
      confidence,
      volume,
      entry_price,
      stop_loss,
      take_profit,
      spread_points,
      (long)retcode,
      retcode_desc,
      note
   );
   FileClose(handle);
  }

bool BuildOrderLevels(int direction, MqlTick &tick, double &entry_price, double &stop_loss, double &take_profit, double &spread_points, string &reason)
  {
   if(direction == 0)
     {
      reason = "hold_signal";
      return false;
     }

   if(!SymbolInfoTick(_Symbol, tick))
     {
      reason = "tick_unavailable";
      return false;
     }

   spread_points = CurrentSpreadPoints(tick);
   if(InpMaxSpreadPoints > 0 && spread_points > (double)InpMaxSpreadPoints)
     {
      reason = "spread_too_wide";
      return false;
     }

   double atr14 = 0.0;
   if(!CopyAtr14(atr14))
     {
      reason = "atr_unavailable";
      return false;
     }

   double minimum_stop_distance = ((double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) + (double)InpStopBufferPoints) * _Point;
   double stop_distance = MathMax(atr14 * InpStopAtrMultiple, minimum_stop_distance);
   if(stop_distance <= 0.0)
     {
      reason = "invalid_stop_distance";
      return false;
     }

   if(direction > 0)
     {
      entry_price = tick.ask;
      stop_loss = NormalizePrice(entry_price - stop_distance);
      take_profit = NormalizePrice(entry_price + stop_distance * InpTakeProfitRR);
     }
   else
     {
      entry_price = tick.bid;
      stop_loss = NormalizePrice(entry_price + stop_distance);
      take_profit = NormalizePrice(entry_price - stop_distance * InpTakeProfitRR);
     }

   if(stop_loss <= 0.0 || take_profit <= 0.0)
     {
      reason = "invalid_trade_levels";
      return false;
     }

   return true;
  }

bool BuildDirectiveOrderLevels(int direction, const TradeDirective &directive, MqlTick &tick, double &entry_price, double &stop_loss, double &take_profit, double &spread_points, string &reason)
  {
   if(direction == 0)
     {
      reason = "hold_signal";
      return false;
     }

   if(!SymbolInfoTick(_Symbol, tick))
     {
      reason = "tick_unavailable";
      return false;
     }

   spread_points = CurrentSpreadPoints(tick);
   if(InpMaxSpreadPoints > 0 && spread_points > (double)InpMaxSpreadPoints)
     {
      reason = "spread_too_wide";
      return false;
     }

   entry_price = (direction > 0 ? tick.ask : tick.bid);

   double minimum_stop_distance = ((double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) + (double)InpStopBufferPoints) * _Point;
   double directive_stop_distance = MathAbs(directive.entry_price - directive.stop_loss);
   double directive_tp_distance = MathAbs((directive.tp2 > 0.0 ? directive.tp2 : directive.tp1) - directive.entry_price);
   double stop_distance = MathMax(directive_stop_distance, minimum_stop_distance);
   double take_profit_distance = MathMax(directive_tp_distance, minimum_stop_distance);
   if(stop_distance <= 0.0 || take_profit_distance <= 0.0)
     {
      reason = "directive_levels_missing";
      return false;
     }

   if(direction > 0)
     {
      stop_loss = NormalizePrice(entry_price - stop_distance);
      take_profit = NormalizePrice(entry_price + take_profit_distance);
     }
   else
     {
      stop_loss = NormalizePrice(entry_price + stop_distance);
      take_profit = NormalizePrice(entry_price - take_profit_distance);
     }

   if(stop_loss <= 0.0 || take_profit <= 0.0)
     {
      reason = "directive_levels_invalid";
      return false;
     }

   return true;
  }

bool MaybeExecuteTrade(datetime bar_time_utc, long predicted_class, const float &probabilities[])
  {
   if(InpValidationMode || !InpEnableDemoTrading)
      return true;

   int direction = PredictionDirection(predicted_class);
   double confidence = MathMax((double)probabilities[0], MathMax((double)probabilities[1], (double)probabilities[2]));

   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !MQLInfoInteger(MQL_TRADE_ALLOWED))
     {
      LogAwaitingCondition(
         bar_time_utc,
         StringFormat(
            "Trade permissions blocked for bar %s: terminal=%d ea=%d",
            TimeToString(bar_time_utc, TIME_DATE | TIME_SECONDS),
            (int)TerminalInfoInteger(TERMINAL_TRADE_ALLOWED),
            (int)MQLInfoInteger(MQL_TRADE_ALLOWED)
         )
      );
      return false;
     }

   TradeDirective directive;
   bool use_directive_levels = false;
   if(InpRequireTradeDirective)
     {
      if(!LoadLatestTradeDirective(directive))
        {
         LogAwaitingCondition(
            bar_time_utc,
            StringFormat(
               "Waiting for trade directive for bar %s: latest directive file could not be read yet.",
               TimeToString(bar_time_utc, TIME_DATE | TIME_SECONDS)
            )
         );
         return false;
        }

      if(directive.time_utc > bar_time_utc)
        {
         LogAwaitingCondition(
            bar_time_utc,
            StringFormat(
               "Waiting for matching directive: bar=%s latest_directive=%s",
               TimeToString(bar_time_utc, TIME_DATE | TIME_SECONDS),
               TimeToString(directive.time_utc, TIME_DATE | TIME_SECONDS)
            )
         );
         return false;
        }
      if(directive.time_utc < bar_time_utc)
        {
         int directive_age_seconds = (int)(bar_time_utc - directive.time_utc);
         if(g_lastConsumedDirectiveUtc == directive.time_utc)
           {
            AppendTradeLog(bar_time_utc, "SKIP", DirectionToString(DirectiveDirection(directive)), directive.directional_confidence, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "directive_already_consumed");
            return true;
           }
         if(InpDirectiveMaxAgeBars > 0)
           {
            int period_seconds = PeriodSeconds(PERIOD_CURRENT);
            if(period_seconds <= 0)
               period_seconds = 60;
            int max_age_seconds = MathMax(period_seconds, 1) * InpDirectiveMaxAgeBars;
            if(directive_age_seconds > max_age_seconds)
              {
               LogAwaitingCondition(
                  bar_time_utc,
                  StringFormat(
                     "Directive is stale for bar=%s latest_directive=%s age_seconds=%d",
                     TimeToString(bar_time_utc, TIME_DATE | TIME_SECONDS),
                     TimeToString(directive.time_utc, TIME_DATE | TIME_SECONDS),
                     directive_age_seconds
                  )
               );
               return false;
              }
           }
         PrintFormat(
            "Using prior directive for bar=%s latest_directive=%s age_seconds=%d",
            TimeToString(bar_time_utc, TIME_DATE | TIME_SECONDS),
            TimeToString(directive.time_utc, TIME_DATE | TIME_SECONDS),
            directive_age_seconds
         );
        }

      direction = DirectiveDirection(directive);
      confidence = MathMax(0.0, MathMin(1.0, directive.directional_confidence));
      if(directive.gate_status != "READY" || directive.paper_status != "SIGNAL_READY" || direction == 0)
        {
         string directive_block_reason = directive.paper_reason_blocked;
         if(directive_block_reason == "")
            directive_block_reason = "trade_directive_blocked";
         string directive_direction_name = DirectionToString(direction);
         if(direction == 0)
            directive_direction_name = "HOLD";
         AppendTradeLog(bar_time_utc, "BLOCK", directive_direction_name, confidence, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", directive_block_reason);
         return true;
        }

      use_directive_levels = true;
     }
   else
     {
      string inferred_direction_name = DirectionToString(direction);
      if(direction == 0)
        {
         AppendTradeLog(bar_time_utc, "SKIP", inferred_direction_name, confidence, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "hold_signal");
         return true;
        }

      if(confidence < InpConfidenceThresh)
        {
         AppendTradeLog(bar_time_utc, "SKIP", inferred_direction_name, confidence, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "below_confidence_threshold");
         return true;
        }
     }

   string direction_name = DirectionToString(direction);
   g_lastAwaitingBarUtc = 0;
   g_lastAwaitingMessage = "";

   if(!IsDemoAccount())
     {
      AppendTradeLog(bar_time_utc, "BLOCK", direction_name, confidence, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "demo_only_guard");
      return true;
     }

   if(!IsTradeModeAllowed(direction))
     {
      AppendTradeLog(bar_time_utc, "BLOCK", direction_name, confidence, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "symbol_trade_mode_blocked");
      return true;
     }

   if(InpSessionTradeLimit > 0 && g_successfulTradeCount >= InpSessionTradeLimit)
     {
      AppendTradeLog(bar_time_utc, "BLOCK", direction_name, confidence, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "session_trade_limit_reached");
      return true;
     }

   ulong position_ticket = 0;
   long position_type = -1;
   int managed_positions = CountManagedPositions(position_ticket, position_type);
   if(managed_positions > 1)
     {
      AppendTradeLog(bar_time_utc, "BLOCK", direction_name, confidence, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "multiple_managed_positions_open");
      return true;
     }

   if(managed_positions == 1)
     {
      int existing_direction = 0;
      if(position_type == POSITION_TYPE_BUY)
         existing_direction = 1;
      else if(position_type == POSITION_TYPE_SELL)
         existing_direction = -1;

      if(existing_direction == direction)
        {
         AppendTradeLog(bar_time_utc, "SKIP", direction_name, confidence, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "existing_same_direction_position");
         return true;
        }

      if(!InpAllowSignalFlip)
        {
         AppendTradeLog(bar_time_utc, "SKIP", direction_name, confidence, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "opposite_position_open");
         return true;
        }

      if(!ExtTrade.PositionClose(position_ticket))
        {
         AppendTradeLog(
            bar_time_utc,
            "CLOSE_REJECT",
            PositionTypeToString(position_type),
            confidence,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            ExtTrade.ResultRetcode(),
            ExtTrade.ResultRetcodeDescription(),
            "flip_before_entry_failed"
         );
         return true;
        }

      AppendTradeLog(
         bar_time_utc,
         "CLOSE",
         PositionTypeToString(position_type),
         confidence,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         ExtTrade.ResultRetcode(),
         ExtTrade.ResultRetcodeDescription(),
         "flip_before_entry"
      );
     }

   MqlTick tick;
   double entry_price = 0.0;
   double stop_loss = 0.0;
   double take_profit = 0.0;
   double breakeven_trigger_price = 0.0;
   double spread_points = 0.0;
   string blocked_reason = "";
   bool levels_ready = false;
   if(use_directive_levels)
      levels_ready = BuildDirectiveOrderLevels(direction, directive, tick, entry_price, stop_loss, take_profit, spread_points, blocked_reason);
   else
      levels_ready = BuildOrderLevels(direction, tick, entry_price, stop_loss, take_profit, spread_points, blocked_reason);
   if(!levels_ready)
     {
      AppendTradeLog(bar_time_utc, "BLOCK", direction_name, confidence, 0.0, entry_price, stop_loss, take_profit, spread_points, 0, "", blocked_reason);
      return true;
     }

   if(use_directive_levels && directive.tp1 > 0.0)
     {
      double tp1_distance = MathAbs(directive.tp1 - directive.entry_price);
      if(direction > 0)
         breakeven_trigger_price = NormalizePrice(entry_price + tp1_distance);
      else
         breakeven_trigger_price = NormalizePrice(entry_price - tp1_distance);
     }
   else
      breakeven_trigger_price = take_profit;

   double requested_volume = InpFixedLotSize;
   bool use_directive_volume = (use_directive_levels && directive.volume_lots > 0.0);
   if(use_directive_volume)
      requested_volume = directive.volume_lots;
   else if(InpUseRiskBasedSizing)
     {
      string sizing_reason = "";
      requested_volume = CalculateRiskBasedVolume(direction, entry_price, stop_loss, sizing_reason);
      if(requested_volume <= 0.0)
        {
         if(sizing_reason == "")
            sizing_reason = "risk_based_sizing_failed";
         AppendTradeLog(bar_time_utc, "BLOCK", direction_name, confidence, 0.0, entry_price, stop_loss, take_profit, spread_points, 0, "", sizing_reason);
         return true;
        }
     }

   if(!use_directive_volume && InpEnableDemoTrading && InpDemoMaxLotSize > 0.0)
      requested_volume = MathMin(requested_volume, InpDemoMaxLotSize);

   double volume = NormalizeVolume(requested_volume);
   if(volume <= 0.0)
     {
      AppendTradeLog(bar_time_utc, "BLOCK", direction_name, confidence, volume, entry_price, stop_loss, take_profit, spread_points, 0, "", "invalid_volume");
      return true;
     }

   ExtTrade.SetExpertMagicNumber(InpMagicNumber);
   ExtTrade.SetDeviationInPoints(InpSlippagePoints);
   ExtTrade.SetAsyncMode(false);
   ExtTrade.SetTypeFillingBySymbol(_Symbol);

   bool submitted = false;
   if(direction > 0)
      submitted = ExtTrade.Buy(volume, _Symbol, 0.0, stop_loss, take_profit, InpTradeComment);
   else
      submitted = ExtTrade.Sell(volume, _Symbol, 0.0, stop_loss, take_profit, InpTradeComment);

   AppendTradeLog(
      bar_time_utc,
      (submitted ? "OPEN" : "OPEN_REJECT"),
      direction_name,
      confidence,
      volume,
      entry_price,
      stop_loss,
      take_profit,
      spread_points,
      ExtTrade.ResultRetcode(),
      ExtTrade.ResultRetcodeDescription(),
      (submitted ? "order_submitted" : "order_failed")
   );

   if(submitted)
     {
      EnsureManagedPositionProtection(bar_time_utc, direction, stop_loss, take_profit);
      if(use_directive_levels)
         g_lastConsumedDirectiveUtc = directive.time_utc;
      g_activeTradeEntry = entry_price;
      g_activeTradeTp1 = breakeven_trigger_price;
      g_activeTradeBreakevenDone = false;
      g_activeTradeDirection = direction;
      g_successfulTradeCount++;
      PrintFormat(
         "Demo trade submitted: %s volume=%.2f entry=%.2f sl=%.2f tp=%.2f confidence=%.4f spread=%.1f successful_trade_count=%d",
         direction_name,
         volume,
         entry_price,
         stop_loss,
         take_profit,
         confidence,
         spread_points,
         g_successfulTradeCount
      );
     }
   else
     {
      PrintFormat(
         "Demo trade rejected: %s retcode=%d (%s)",
         direction_name,
         (long)ExtTrade.ResultRetcode(),
         ExtTrade.ResultRetcodeDescription()
      );
     }
   return true;
  }

void AppendSignalLog(datetime bar_time_utc, long predicted_class, const float &probabilities[], const float &features[], int validation_status)
  {
   if(!InpLogSignals)
      return;

   FolderCreate("logs");
   int handle = FileOpen(LOG_FILE, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_SHARE_READ, ',');
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
   int handle = FileOpen(VALIDATION_FILE, FILE_READ | FILE_CSV | FILE_ANSI, ',');
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
   Print("--- XAUUSD AI Bot v", BOT_VERSION, " Initializing ---");

   if(InpEnableDemoTrading && !TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
      Print("Auto-trading is currently disabled in the terminal. The EA will stay idle until you enable it.");
   if(InpEnableDemoTrading && !MQLInfoInteger(MQL_TRADE_ALLOWED))
      Print("EA-level trading permission is disabled in the Common tab. Enable 'Allow Algo Trading' for this chart.");

   if(InpEnableDemoTrading && !IsDemoAccount())
     {
      Print("Demo-account broker execution is enforced, but the connected account is not a demo account.");
      return INIT_FAILED;
     }

   if(InpEnableDemoTrading && !InpDemoOnly)
      Print("InpDemoOnly=false was ignored. Broker execution remains hard-locked to demo accounts only.");
   if(InpEnableDemoTrading && !InpUseRiskBasedSizing && InpDemoMaxLotSize > 0.0 && InpFixedLotSize > InpDemoMaxLotSize)
      PrintFormat("InpFixedLotSize=%.2f was capped. Demo broker execution cannot exceed %.2f lots.", InpFixedLotSize, InpDemoMaxLotSize);

   string resolved_model_name = "";
   if(!InpRequireTradeDirective)
     {
      resolved_model_name = ResolveModelName();
      ExtOnnxHandle = OnnxCreate(resolved_model_name, ONNX_DEFAULT);
      if(ExtOnnxHandle == INVALID_HANDLE)
        {
         PrintFormat("Failed to load ONNX model '%s'. Error=%d", resolved_model_name, GetLastError());
         return INIT_FAILED;
        }

      long input_shape[] = {1, FEATURE_COUNT};
      if(!OnnxSetInputShape(ExtOnnxHandle, 0, input_shape))
         Print("Warning: OnnxSetInputShape failed. Error: ", GetLastError());
     }

   g_hATR14 = iATR(_Symbol, PERIOD_CURRENT, 14);
   if(g_hATR14 == INVALID_HANDLE)
     {
      Print("ATR handle initialization failed.");
      return INIT_FAILED;
     }

   if(!ExtFeatureEngine.Init(_Symbol, PERIOD_CURRENT))
     {
      Print("Feature engine initialization failed.");
      return INIT_FAILED;
     }

   FolderCreate("logs");
   WriteAccountSnapshot();
   EventSetTimer(1);
   ExtTrade.SetExpertMagicNumber(InpMagicNumber);
   ExtTrade.SetDeviationInPoints(InpSlippagePoints);
   ExtTrade.SetAsyncMode(false);
   ExtTrade.SetTypeFillingBySymbol(_Symbol);
   Print("Validation mode: ", (InpValidationMode ? "ON" : "OFF"));
   if(InpRequireTradeDirective)
      Print("Trade directive mode is ON: local ONNX inference is bypassed and MT5 mirrors the paper-trade directive.");
   else
      Print("Model loaded: ", resolved_model_name);
   if(!InpValidationMode && !InpEnableDemoTrading)
      Print("Live signal mode is ON, but demo trade execution is disabled.");
   if(!InpValidationMode && InpEnableDemoTrading)
      Print("Demo trade execution is armed with demo-only safety checks and paper-directive gating.");
   if(InpEnableDemoTrading && InpUseRiskBasedSizing)
      PrintFormat("Risk-based sizing is ON: %.2f%% of current account balance per trade (demo cap %.2f lots).", InpRiskPerTradePercent, InpDemoMaxLotSize);
   if(InpEnableDemoTrading && InpSessionTradeLimit > 0)
      PrintFormat("This EA session is limited to %d successful demo trade(s).", InpSessionTradeLimit);
   if(InpEnableDemoTrading && InpRequireTradeDirective && !FileIsExist(TRADE_DIRECTIVE_FILE))
      Print("Trade directive file is not available yet. Keep the MT5 research worker running; the EA will wait for the paper-trade directive.");
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ExtFeatureEngine.Release();
   if(g_hATR14 != INVALID_HANDLE)
     {
      IndicatorRelease(g_hATR14);
      g_hATR14 = INVALID_HANDLE;
     }
   if(ExtOnnxHandle != INVALID_HANDLE)
     {
      OnnxRelease(ExtOnnxHandle);
      ExtOnnxHandle = INVALID_HANDLE;
     }
  }

void OnTick()
  {
   WriteAccountSnapshot();
   ManageOpenPositionBreakeven();
   datetime current_bar_open = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_bar_open != g_lastBarOpenTime)
     {
      g_lastBarOpenTime = current_bar_open;
      g_hasPendingTrade = false;

      datetime closed_bar_server_time = iTime(_Symbol, PERIOD_CURRENT, 1);
      datetime closed_bar_utc = ToUtc(closed_bar_server_time);
      if(!InpRequireTradeDirective && !IsOverlapBar(closed_bar_utc))
         return;

      long prediction_class[1] = {1};
      float probabilities[3] = {0.0f, 1.0f, 0.0f};
      if(!InpRequireTradeDirective)
        {
         float features[FEATURE_COUNT];
         if(!ExtFeatureEngine.ComputeFeatures(1, features))
           {
            Print("Feature computation failed.");
            return;
           }

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
            "UTC=%s signal=%s probs=[%.4f, %.4f, %.4f] confidence=%.4f validation=%s",
            TimeToString(closed_bar_utc, TIME_DATE | TIME_SECONDS),
            PredictionLabel(prediction_class[0]),
            probabilities[0],
            probabilities[1],
            probabilities[2],
            MathMax(probabilities[0], MathMax(probabilities[1], probabilities[2])),
            ValidationStatusToString(validation_status)
         );
        }

      g_pendingTradeBarUtc = closed_bar_utc;
      g_pendingPredictedClass = prediction_class[0];
      for(int index = 0; index < 3; index++)
         g_pendingProbabilities[index] = probabilities[index];
      g_hasPendingTrade = true;
     }

   if(g_hasPendingTrade && MaybeExecuteTrade(g_pendingTradeBarUtc, g_pendingPredictedClass, g_pendingProbabilities))
      g_hasPendingTrade = false;
  }

void OnTimer()
  {
   WriteAccountSnapshot();
   ManageOpenPositionBreakeven();
  }
//+------------------------------------------------------------------+
