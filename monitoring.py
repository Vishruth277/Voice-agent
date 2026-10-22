import os
import time
import psutil

# Configuration settings
POLL_INTERVAL = 0.5  # Check system usage every 500 milliseconds
CPU_SPIKE_THRESHOLD = 15.0  # Trigger display only if total CPU drops or spikes past 15%

def find_process_by_name(name):
    """Finds all active PIDs matching a specific execution name."""
    pids = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if name matches process name or python script argument execution
            if name.lower() in proc.info['name'].lower():
                pids.append(proc.info['pid'])
            elif proc.info['cmdline'] and any(name.lower() in cmd.lower() for cmd in proc.info['cmdline']):
                pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pids

if __name__ == "__main__":
    print("="*60)
    print("🛰️ STANDALONE BACKGROUND PERFORMANCE MONITOR ACTIVE")
    print("="*60)
    print(f"Polling system state every {POLL_INTERVAL}s... Waiting for activity.")
    print("Run 'python3 assistant.py' in your other terminal window now.\n")

    # Initialize CPU baseline tracking values
    psutil.cpu_percent(interval=None)

    while True:
        try:
            # 1. Dynamically discover process IDs across the operating system layer
            assistant_pids = find_process_by_name("assistant.py")
            ollama_pids = find_process_by_name("ollama")

            # Combine them for tracking collective compute load metrics
            all_tracked_pids = assistant_pids + ollama_pids

            if not all_tracked_pids:
                # System is completely idle / assistant script is turned off
                time.sleep(1.0)
                continue

            # 2. Sample global CPU load over the interval window
            current_cpu = psutil.cpu_percent(interval=POLL_INTERVAL)

            # 3. If a CPU load spike is noticed, it means an inference event occurred
            if current_cpu >= CPU_SPIKE_THRESHOLD:
                print("\n" + "="*50)
                print("📊 LIVE COMPUTE MONITOR TELEMETRY")
                print("="*50)
                
                # Fetch script memory allocations safely
                assistant_ram = 0.0
                for pid in assistant_pids:
                    try:
                        assistant_ram += psutil.Process(pid).memory_info().rss / (1024 * 1024)
                    except Exception: pass

                # Fetch underlying LLM model server allocations safely
                ollama_ram = 0.0
                for pid in ollama_pids:
                    try:
                        ollama_ram += psutil.Process(pid).memory_info().rss / (1024 * 1024)
                    except Exception: pass

                # Display targeted metric breakdown boxes
                print(f"⚙️ Live System CPU Load : {current_cpu}% utilization")
                print(f"🐍 assistant.py Memory  : {assistant_ram:.1f} MB RAM active")
                print(f"🧠 Ollama Engine Memory : {ollama_ram:.1f} MB RAM active")
                print(f"📦 Total Combined Memory: {(assistant_ram + ollama_ram):.1f} MB RAM footprint")
                print(f"💾 Total Available RAM  : {psutil.virtual_memory().available / (1024*1024*1024):.2f} GB remaining")
                print("="*50)
                
                # Cool-down delay to prevent printing duplicates during the same sentence generation turn
                time.sleep(2.0)

        except KeyboardInterrupt:
            print("\nMonitoring terminated cleanly. Goodbye.")
            break
        except Exception as e:
            time.sleep(1.0)

