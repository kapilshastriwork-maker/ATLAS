import multiprocessing
import sys
import os


def start_server(module_name, port):
    import uvicorn
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    uvicorn.run(f"mcp_servers.{module_name}:app", host="0.0.0.0", port=port, log_level="warning")


def run_all_servers():
    servers = [
        ("handoff_server", 8001),
        ("medication_server", 8002),
        ("prior_auth_server", 8003),
        ("navigator_server", 8004),
        ("sdoh_server", 8005),
    ]
    
    print("=" * 60)
    print("ATLAS Phase 2 - MCP Superpowers")
    print("=" * 60)
    print("\nStarting all 5 MCP servers...\n")
    
    for name, port in servers:
        print(f"  {name:20} → http://localhost:{port}")
    
    print("\n" + "=" * 60)
    print("Server ready! Press Ctrl+C to stop all servers.")
    print("=" * 60 + "\n")
    
    processes = []
    
    try:
        for module, port in servers:
            p = multiprocessing.Process(target=start_server, args=(module, port))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        print("\n\nShutting down all servers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        print("All servers stopped.")


if __name__ == "__main__":
    run_all_servers()