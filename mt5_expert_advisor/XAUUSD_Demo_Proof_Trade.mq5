//+------------------------------------------------------------------+
//|                                      XAUUSD_Demo_Proof_Trade.mq5 |
//|                                                       Andy Warui |
//+------------------------------------------------------------------+
#property copyright "Andy Warui"
#property link      "https://github.com/andywarui/xauusd-ai-trading-bot"
#property version   "1.000"
#property script_show_inputs
#property description "One-shot demo proof trade for XAUUSD. Opens 0.01 lot max, waits 60 seconds, then closes."

#include <Trade/Trade.mqh>

input double   InpLotSize         = 0.01;
input int      InpHoldSeconds     = 60;
input bool     InpUseBuyOrder     = true;
input int      InpMagicNumber     = 202699;
input int      InpSlippagePoints  = 30;
input string   InpTradeComment    = "XAUUSD_AI_Proof";

#define DEMO_MAX_LOT_SIZE 0.01

CTrade ExtTrade;

bool IsDemoAccount()
  {
   return ((ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_TRADE_MODE_DEMO);
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

ulong FindManagedPositionTicket()
  {
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
      return current_ticket;
     }
   return 0;
  }

void OnStart()
  {
   Print("--- XAUUSD Demo Proof Trade starting ---");

   if(!IsDemoAccount())
     {
      Print("Proof trade is demo-only. Aborting because the connected account is not a demo account.");
      return;
     }

   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !MQLInfoInteger(MQL_TRADE_ALLOWED))
     {
      Print("Proof trade aborted because automated trading is not enabled.");
      return;
     }

   if(FindManagedPositionTicket() != 0)
     {
      Print("Proof trade aborted because a proof-trade position is already open for this symbol/magic.");
      return;
     }

   double requested_volume = MathMin(InpLotSize, DEMO_MAX_LOT_SIZE);
   double volume = NormalizeVolume(requested_volume);
   if(volume <= 0.0)
     {
      Print("Proof trade aborted because the normalized volume is invalid.");
      return;
     }

   ExtTrade.SetExpertMagicNumber(InpMagicNumber);
   ExtTrade.SetDeviationInPoints(InpSlippagePoints);
   ExtTrade.SetAsyncMode(false);
   ExtTrade.SetTypeFillingBySymbol(_Symbol);

   bool submitted = false;
   if(InpUseBuyOrder)
      submitted = ExtTrade.Buy(volume, _Symbol, 0.0, 0.0, 0.0, InpTradeComment);
   else
      submitted = ExtTrade.Sell(volume, _Symbol, 0.0, 0.0, 0.0, InpTradeComment);

   if(!submitted)
     {
      PrintFormat(
         "Proof trade open failed. retcode=%d (%s)",
         (long)ExtTrade.ResultRetcode(),
         ExtTrade.ResultRetcodeDescription()
      );
      return;
     }

   ulong position_ticket = FindManagedPositionTicket();
   PrintFormat(
      "Proof trade opened successfully. ticket=%I64u volume=%.2f side=%s hold_seconds=%d",
      position_ticket,
      volume,
      (InpUseBuyOrder ? "BUY" : "SELL"),
      InpHoldSeconds
   );

   int hold_seconds = MathMax(1, InpHoldSeconds);
   for(int second = 0; second < hold_seconds && !IsStopped(); second++)
      Sleep(1000);

   position_ticket = FindManagedPositionTicket();
   if(position_ticket == 0)
     {
      Print("Proof trade close skipped because the position is no longer open.");
      return;
     }

   if(!ExtTrade.PositionClose(position_ticket))
     {
      PrintFormat(
         "Proof trade close failed. ticket=%I64u retcode=%d (%s)",
         position_ticket,
         (long)ExtTrade.ResultRetcode(),
         ExtTrade.ResultRetcodeDescription()
      );
      return;
     }

   PrintFormat(
      "Proof trade closed successfully. ticket=%I64u retcode=%d (%s)",
      position_ticket,
      (long)ExtTrade.ResultRetcode(),
      ExtTrade.ResultRetcodeDescription()
   );
  }
//+------------------------------------------------------------------+
