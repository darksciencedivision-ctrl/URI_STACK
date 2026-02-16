# URI / ORCHESTRA / PRAXIS Stack (Offline File-Bus)

Local-first conversational AI research platform.

## Components
- **URI**: Web UI + REST control plane (submits jobs, shows artifacts, tails logs)
- **ORCHESTRA**: File-bus job runner (single + RRR modes), writes reducer outputs + artifacts
- **PRAXIS**: Authority ingestor (validates reducer schema, appends canonical state)

## Run
### 1) Start PRAXIS
\\\powershell
cd .\ai_control\PRAXIS
python .\praxis_ingestor.py
\\\

### 2) Start ORCHESTRA
\\\powershell
cd .\ai_control\ORCHESTRA
python .\orchestra_runner.py
\\\

### 3) Start URI
\\\powershell
cd .\URI
python .\uri.py
\\\

Open: http://127.0.0.1:8088
