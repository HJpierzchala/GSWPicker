GSWPicker/
├── modules/                    # Core processing modules
│   ├── classes.py             # Initializes processing per station and MODE (MAD, SLOPE, W-TEST)
│   ├── detect.py              # Core module for S-wave picking
│   ├── output.py              # Functions for generating tabular (.csv) results
│   ├── parallel.py            # Enables parallel processing via concurrent.futures
│   ├── prep.py                # Handles input data, filtering, coordinate transformations
│   ├── time.py                # Time conversions (GPST, UTC, MJD), leap second handling
│   ├── UI.py                  # GUI-related functionality
│   └── preinspect.py          # Tools for visualizing raw velocity noise and spectral content
│
├── results/                   # Output results go here
│   └── test/                  # Example output
│
├── vel_data/                  # Input GNSS data
│   ├── Test/
│   └── Test_kin/
│
├── Config/                    # Configuration and installation files
│   ├── requirements.txt
│   ├── parameters.toml
│   └── leap_seconds_table.csv
│
├── docs/                      # Documentation
│   ├── User_manual.pdf
│   └── User_manual.md
│
├── GSW_GUI.py                 # Launches the GUI
├── mainGUI.py                 # Handles terminal visual output
├── make_tree.py               # Script to create required folder structure
├── install.sh                 # Bash script for auto-installation
├── setup.py                   # Python installer script
└── pyproject.toml             # Metadata and dependencies for package managers
