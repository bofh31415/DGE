from main import DGELab

if __name__ == "__main__":
    lab = DGELab()
    # Bypass menu, run chain directly
    # Need to instantiate optimizer first as in main.py reset_model
    lab.reset_model() 
    lab.run_dge_validation_chain()
