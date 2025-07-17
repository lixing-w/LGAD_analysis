This project analyzes databases containing IV and CV scans of LGADs.

To run the project:
    1)  Specify the root directory to target database in utils.py, by setting
        DATABASE_DIR. Each database should contain sensor folders, which 
        contain scans or folders of scans. All sensor folders are assumed to be 
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
        which analyzes all available IV scans in the database. For more info,
        type the following command:
            python3 analyze.py --help
        which prints out available options to use. 
    3)  View plots in your sensor folders.
    4)  Further configure your analysis with data_config.txt and sensor_config.txt
        at your database root.

Structure:
    analyze.py      - main script for analysis and plotting
    Sensor.py       - manages the Sensor class
    utils.py        - contains database root and miscellaneous functions

Dev logs:
    Jun 19 (Lixing):
        - added compatibility of .iv files 
        - added Sensor class to improve readability
        - added more documentation 
        - bug fixes
    
    Jun 30 (Lixing):
        - added more data 
        - significant performance optimization to find_threshold
        - implemented RANSAC as an alternative method to fit lines 
        - implemented sub-interval interpolation for more accurate breakdown 
        voltage estimation
        - switching from max() to + for measurement uncertainty
        - bug fixes
        - improved readability

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
        - finished config system for sensor. To view info or configure 
        sensors, go to sensor_config.txt at root of your database. Each 
        database has one sensor_config.txt of its own.
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

        Format:
            [command] [argument1:value1] [argument2:value2] ...
            One expression Per line.
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
            1. N - name of the sensor, must be specified in every expression
            2. T - temperature
            3. R - regex pattern string
            4. D - date, in dd/mm/yyyy
            5. F - relative path to scan
            6. DEP - voltage after which ransac is applied (for IV scan only)
            7. RT - ramp type, either -1 (down), 1 (up), 0 (none)
            Note: All except DEP can be used after DR and SPEC. Only DEP and 
            RT can be used after SET (for now).
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