#property strict
#property version   "1.00"
#property description "Exports recent XAUUSD bars with volume into MQL5/Files/xauusd_mt5_live.csv for the local Python research pipeline."

input string ExportFile = "xauusd_mt5_live.csv";
input string AccountSnapshotFile = "config\\mt5_account_snapshot.csv";
input string ExpectedSymbol = "XAUUSD";
input ENUM_TIMEFRAMES ExportTimeframe = PERIOD_M1;
input int BarsToExport = 5000;
input int RefreshSeconds = 5;
input bool UseRealVolumeIfAvailable = false;
input int IntraBarRefreshSeconds = 3;

datetime g_last_exported_bar = 0;
datetime g_last_export_wallclock = 0;
bool g_symbol_warning_printed = false;
double g_last_exported_close = 0.0;
long g_last_exported_volume = -1;

bool SymbolMatchesExpected()
{
   string chart_symbol = _Symbol;
   string expected_symbol = ExpectedSymbol;
   StringToUpper(chart_symbol);
   StringToUpper(expected_symbol);
   return chart_symbol == expected_symbol || StringFind(chart_symbol, expected_symbol) >= 0;
}

bool EnsureExpectedSymbol()
{
   if(SymbolMatchesExpected())
      return true;

   if(!g_symbol_warning_printed)
   {
      Print("Exporter: attached to ", _Symbol, " but ExpectedSymbol is ", ExpectedSymbol, ". Export skipped until the EA is attached to the correct XAUUSD chart.");
      g_symbol_warning_printed = true;
   }
   return false;
}

void ExportAccountSnapshot()
{
   FolderCreate("config");

   int handle = FileOpen(AccountSnapshotFile, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("Exporter: Account snapshot FileOpen failed for ", AccountSnapshotFile, " error=", GetLastError());
      return;
   }

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   long login = AccountInfoInteger(ACCOUNT_LOGIN);
   long leverage = AccountInfoInteger(ACCOUNT_LEVERAGE);
   string server = AccountInfoString(ACCOUNT_SERVER);
   string currency = AccountInfoString(ACCOUNT_CURRENCY);
   double contract_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   double volume_min = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double volume_max = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

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
      (string)login,
      server,
      currency,
      DoubleToString(balance, 2),
      DoubleToString(equity, 2),
      (string)leverage,
      DoubleToString(contract_size, 2),
      DoubleToString(volume_min, 2),
      DoubleToString(volume_max, 2),
      DoubleToString(volume_step, 2)
   );

   FileClose(handle);
}

void ExportBars()
{
   if(!EnsureExpectedSymbol())
      return;

   MqlRates rates[];
   int copied = CopyRates(_Symbol, ExportTimeframe, 0, BarsToExport, rates);
   if(copied <= 0)
   {
      Print("Exporter: CopyRates failed for ", _Symbol, " error=", GetLastError());
      return;
   }

   ArraySetAsSeries(rates, false);

   int handle = FileOpen(ExportFile, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("Exporter: FileOpen failed for ", ExportFile, " error=", GetLastError());
      return;
   }

   FileWrite(handle, "time", "open", "high", "low", "close", "volume");
   for(int i = 0; i < copied; i++)
   {
      long volume_value = (long)rates[i].tick_volume;
      if(UseRealVolumeIfAvailable && rates[i].real_volume > 0)
         volume_value = (long)rates[i].real_volume;
      string volume_text = (string)volume_value;

      FileWrite(
         handle,
         TimeToString(rates[i].time, TIME_DATE | TIME_SECONDS),
         DoubleToString(rates[i].open, _Digits),
         DoubleToString(rates[i].high, _Digits),
         DoubleToString(rates[i].low, _Digits),
         DoubleToString(rates[i].close, _Digits),
         volume_text
      );
   }

   FileClose(handle);
   ExportAccountSnapshot();
   g_last_exported_bar = rates[copied - 1].time;
   g_last_export_wallclock = TimeCurrent();
   g_last_exported_close = rates[copied - 1].close;
   g_last_exported_volume = (long)rates[copied - 1].tick_volume;
   if(UseRealVolumeIfAvailable && rates[copied - 1].real_volume > 0)
      g_last_exported_volume = (long)rates[copied - 1].real_volume;
   Print("Exporter: wrote ", copied, " bars to ", ExportFile, " for ", _Symbol, " ", EnumToString(ExportTimeframe));
}

void MaybeExport()
{
   if(!EnsureExpectedSymbol())
      return;

   datetime latest_bar = iTime(_Symbol, ExportTimeframe, 0);
   if(latest_bar == 0)
      return;

   double latest_close = iClose(_Symbol, ExportTimeframe, 0);
   long latest_volume = (long)iVolume(_Symbol, ExportTimeframe, 0);
   bool new_bar_started = (g_last_exported_bar == 0 || latest_bar != g_last_exported_bar);
   bool intrabar_state_changed = (latest_close != g_last_exported_close || latest_volume != g_last_exported_volume);
   bool refresh_due = (g_last_export_wallclock == 0 || (TimeCurrent() - g_last_export_wallclock) >= MathMax(1, IntraBarRefreshSeconds));

   if(new_bar_started || (intrabar_state_changed && refresh_due))
      ExportBars();
}

int OnInit()
{
   EventSetTimer((int)MathMax(5, RefreshSeconds));
   ExportBars();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTick()
{
   MaybeExport();
}

void OnTimer()
{
   MaybeExport();
}
