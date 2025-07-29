This project analyzes IV and CV scans of LGADs.

Workflow and Purpose:
    We want to study how environmental conditions might affect performance 
    metrics of LGAD sensors. We primarily focus on IV scans, which tell 
    us the breakdown voltage, depletion voltage, and importantly how the 
    leakage current behaves under different reverse bias voltages.

    We first use Autoencoders to compress IV curves. An autoencoder has 
    two parts: encoder and decoder. The encoder learns to compress data 
    efficiently (the compressed form is called a latent vector), and the 
    decoder learns to reconstruct the data from the latent vector. Then, we 
    study correlations between different environmental variables and 
    different entries of the latent vector. Finally, we map the environmental 
    variables to performance metrics via latent vectors. 
    
    Using autoencoders gives us the advantage of preserving more useful and 
    detailed information from IV curves to construct correlation. They 
    extract whatever information essential to the curves, this is more information 
    than what "usual" line-fitting methods can provide. Also, autoencoders 
    can convert variable-length IV sequences to a fixed-length latent vector, 
    which is convenient for further analysis. 

    However, as of now, the autoencoders have a simple architecture, and 
    explanability of the network may not be promising. Plus, we don't have 
    too many IV scans to learn (e.g. only around 402 good scans in ./data/ivcvscans); 
    we lack information on humidity and measurement duration of many scans; 
    we lack information about the architecture and parameters (such as doping 
    concentration) of the LGADs themselves. Some changes were made to some same 
    LGAD between scans, and we have no good way of quantifying these changes 
    other than knowing the scans took place on different dates.
    The data is not ideal, still, we will see what correlations we can
    explore between environment variables and IV behaviors.

    The autoencoders have latent dimension of 16 for now. After experimentation, 
    we found that any number lower than this caused the model to reconstruct 
    the correct shape but the curves were shifted in some region for many curves, 
    which means the model cannot compress the data to fewer than 16 numbers.
    The dimension could be smaller if we had a more complicated network, 
    but data deficiency suggests we use a small network.

Structure of the project:
    Scripts:
    analyze.py    - script for analysis and plotting with non-ML methods
    utils.py      - lib containing constants, useful functions, and Sensor class
    dataset.py    - defines several datasets for ML
    preprocess.py - lib containing functions to prepare the data for ML
    model.py      - defines pytorch models
    train_autoencoder.py - trains autoencoder to compress IV curve
    visualize_latent.py  - visualize the latent space w/ environmental variables

    Folders:
    archive_code  - some old code created before Summer 2025, not needed for now
    autoencoder_model - stores trained autoencoders
    data          - contains all databases
    miscellaneous - miscellaneous files and data

Running the project:
    1)  Specify the root directory to target database in utils.py, by setting
        DATABASE_DIR. This constant is share across many scripts of the system. 
        If you want to perform any operation on data in a new database, always 
        remeber to change DATABASE_DIR before running the code. Not doing so 
        might lead to undefined behavior.

        All databases should be in folder ./data at the project 
        root. Each database should contain sensor folders, each of which 
        contains scans or folders of scans. All sensor folders are assumed to be 
        titled by the sensors' name.

        Example 1:
            DATABASE_DIR 
                -> Sensor 1
                    -> Date 1
                        -> IV or CV scans...
                    -> Date 2
                        -> IV or CV scans...
                -> Sensor 2
                ...
        Example 2:
            DATABASE_DIR
                -> Sensor 1
                    -> IV or CV scans..
                ...

    2)  To start with, one can run the following command:
            python3 analyze.py --iv 
        which uses a modified version of RANSAC and other non-ML methods 
        to analyze all available IV scans in the database, estimating 
        breakdown voltages and depletion voltages, and plotting related figures. 
        Type the following command for help:
            python3 analyze.py --help

    3)  View plots generated in your sensor folders.
    4)  Further configure your analysis with data_config.txt and sensor_config.txt
        at your database root.
    
    5) Check, train, and modify autoencoders in model.py and train_autoencoder.py
    6) Visualize latent space of the database by running visualize_latent.py

Config Systems:
    At the root of each database, there are two files, data_config.txt 
    and sensor_config.txt, which you can modify to manage data.
    
    sensor_config.txt is where you check data points that analyze.py 
    extracted and set sensor-specific preference for analysis. 

    data_config.txt is where you tell the entire system to ignore some 
    scan (but not delete it), ignore some parts of a scan (maybe cuz it's 
    broken), and override related parameters of a scan. It comes with a 
    dedicated script language.

    Format:
        [command] [argument1:value1] [argument2:value2] ...
        One expression per line.
        Comments can be added anywhere after a '#'.
    Keywords: 
        1. DR...            - ignore specified scans
        2. SPEC...SET...    - specify scans, and set their attributes
        3. ~                - indicate range, used with temperatures and dates
            a) X~Y          - between X and Y (inclusive)
            b) X~           - at least X 
            c) ~X           - at most X
            d) X            - exactly X
    Supported arguments: 
        1. N - name of the sensor, must be specified in EVERY expression
        2. T - temperature
        3. R - regex pattern string
        4. D - date, in dd/mm/yyyy
        5. F - relative path to scan
        6. DEP - voltage after which a line is fit (for IV scan only)
        7. RT - ramp type, either -1 (down), 1 (up), 0 (none)
        Note: All except DEP can be used after DR and SPEC. Only DEP and 
        RT can be used after SET (for now).
        8. MAX - voltage after which data is completely ignored (CV scan only)
    Notes:
        The expressions are parsed top-down, and the first matching expression
        matches. So you may need to put strict expressions above looser
        ones.
    Examples:
        DR N:AC_W3096 T:~-20 
            => ignore AC_W3096 scans whose temperatures are at most -20C
        DR N:DC_W3045 T:100~120 
            => ignore DC_W3045 scans whose temperatures are between 100C 
            and 120C inclusive
        DR N:AC_W3096 T:-20
            => ignore AC_W3096 scans whose temperature is exactly -20C
        DR N:DC_W3045 T:~-20 RT:-1 D:27/10/2023 
            => ignore DC_W3045 scans whose temperatures are at most -20C, 
            AND ramp type is down, AND date is exactly on 27/10/2023
        DR N:DC_W3045 R:RoomTemp D:27/10/2023~28/10/2023
            => ignore DC_W3045 scans whose file name (at least partially)
            matches with regex "RoomTemp", AND whose dates are between 
            27/10/2023 and 28/10/2023 inclusive
        SPEC N:HPK_LGAD_3_1_6 D:16/07/2021 SET DEP:110 
            => when analyzing HPK_LGAD_3_1_6 scans on date 16/07/2021, 
            set depletion voltage to 110V to ignore bad IV scan data at 
            low voltages
        SPEC N:HPK_LGAD_3_1_6 D:16/07/2021 SET DEP:110 RT:-1
            => when analyzing HPK_LGAD_3_1_6 scans on date 16/07/2021, 
            set depletion voltage to 110V, AND set ramp type to down, 
            regardless the original ramp type.
            => ramp type needs not to be specified if it is given in and 
            automatically parsed from the scan file.

TODO:
    analyze.py:
        1. Implement analyze_file_cv(), a function that analyzes 
           CV scans specified by paths, supporting --file used with --cv.
        2. Fixing plot_humidity_scans(). This function is from 
           older versions of the code and has not been updated yet.
        3. Fixing find_threshold(). For a given sensor, this function 
           finds the bd_thresh, used to determine breakdown in ransac(), that minimizes 
           average uncertainty of estimation. Since higher bd_thresh leads to 
           higher estimated breakdown and vice versa, we decided to use 0.5 unanumously 
           for all sensors for now. But we might need its functionality in the future.

Dev logs:
    Jun 19 (Lixing):
        - added compatibility of .iv files 
        - added Sensor class
        - added documentation and bug fixes
    
    Jun 30 (Lixing):
        - added more data 
        - performance optimization to find_threshold
        - implemented RANSAC as an alternative method to fit lines 
        - implemented sub-interval linear interpolation for breakdown estimation
        - bug fixes

    Jul 6 (Lixing):
        - updated breakdown distribution to use histograms of weights, and 
          showed RMSE curve on the same plot
        - depletion voltage estimation with .cv scans
        - switched to new minimum uncertainty: d/sqrt(N), where 
          d is measurement interval, N is # of data pts used
        - bug fixes and readability improvements

    Jul 12 (Lixing):
        - updated breakdown distribution to also include histograms of 
          frequency, to indicate how often a breakdown point is proposed 
          by RANSAC
        - started the design of a new sensor config system that reads and writes 
          information about sensors from and to a common place on the disk
        - reformatted the code structure; moved miscellaneous functions to 
          utils.py. The main script is now called analyze.py.
        - bug fixes; fixed nan issues with calculating mse in ransac() and 
          with plotting rmse with plt.plot

    Jul 13 (Lixing):
        - rebuilt entirely IV analysis code written by Trevor, achieving 
          all original functionalities, removing all hard-coded stuff, 
          greatly improving readability
        - finished config system for sensor. 
        - started design of a secondary config system for scan files that 
          help specify special ways to process certain data, as a replacement 
          to the originally hard-coded method
        - users can now interact with analyze.py via command line arguments
          that allow us to:
            1. analyze only IV scans or CV scans or both
            2. analyze scans of just one or more particular sensors
            3. specify the current type used for IV analysis (pad, gr, total)
            4. clear all plots generated
            5. analyze specific scans by providing paths to scans (not done yet,
            still cooking)
            6. you can view available commands by typing:
                python3 analyze.py --help

    Jul 14 (Lixing):
        - ran the code on IVCV_UNIGE_STRANGE_FEATURES
        - finished the new config system for scans, which comes with a 
          script language I came up with. This system allows us to 1. ignore
          specified bad scans for analysis 2. set particular attributes for 
          analyzing a scan, (depletion volt and ramp type). The language can 
          be expanded to work with more attributes if needed. To edit, go to 
          data_config.txt. Each database has its own data_config.txt at its root.
        - Note: there's not yet a syntax checking for config expressions. 
          be careful.

    Jul 23 (Lixing):
        - added MAX argument to data config system
        - tried sensor-specific baseline MLP model to predict entire IV curve 
          based on temperature alone (DC_W3058 in ./ivcvscans)

    Jul 26 (Lixing) 
        - implemented depletion voltage estimation on IV curve; this algorithm
          is subject to change due to accuracy issues
        - improved from standard RANSAC of using 2 pts for fitting to using 33% 
          of the pts to fit; resolving issue of RANSAC overestimating 
          breakdown in certain cases; improving robustness in noisy data
    
    Jul 27 (Lixing)
        - finished model architecture of an autoencoder and determined the loss 
          function and appropriate latent dimension to use.
        - trained on 144 (unignored) IV scans of DC_W3058 in ./ivcvscans; 
          achieving good reconstruction results with RMSE of 0.0425 on traing 
          data after 292 epochs
        - consolidated Sensor class to untils.py; reformatted the code
        - NumPy style documentation of functions
    
    Jul 28 (Lixing)
        - trained the autoencoder on an entire dataset (ivcvscans) 
          consisting of 402 (unignored) IV scans from 29 sensors,
          achieving good reconstruction results with RMSE of 0.518 on training 
          data after 275 epochs
        - created visualize_latent.py to plot latent space of the ivcvscans 
          dataset, labeled with environmental variables 
        - bug fixes: inf issues in log of current
        - bug fixes: plt not saving plots after plt.show()
        - suppressed numpy invalid value and division by zero err when taking 
          log of leakage current; we removed NaNs and Infs afterwards
        - reformatted README