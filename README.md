The analysis script is plot_temp_scan.py

To run it all you need to do is create a conda environment with pyplot, numpy, and scipy and run "python plot_temp_scan.py"

This runs the analysis for W3045, W3056, and the AC-LGAD for the thresholds defined in line 470. One can comment this line and uncomment 469 to run the full code that indentifies the ideal threshold for each sensor


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
        - switching from min() to + for measurement uncertainty
        - bug fixes
        - improved readability

