# TML-26 

## Assignment 1 Remarks

- The **main submission file**, "task_template.py" is located in the location. "Assignment_1/task_template.py". This is the file that we used to submit to the leaderboard

- The pt_files downloaded using the wget function are located in the folder "Assignment_1/pt_files"

- We also performed various experiments trying to implement the papers provided in the assignment sheet. You can find those misc. experiments files in the folder: "misc_experiments". 

- The final submission files, which are handed-in via CMS are locatted in the folder "Assignment_1/submission". 

## Assignment 2 Remarks

1. The code files pertaining to assignment 1 are located in the folder "ass_2/"
2. Inside the folder you will find the following .py files and folder: 
   - submission.py - File used to submit submission.csv in the submissions folder: 
   - task_template_v0.py - File with the first approach i.e. Parameter space similarity
   - task_template_kl_div.py - File with the 2nd approach, i.e. Functional space similarity
   - task_template_adversedial_fp.py - File with the 3rd, untried approach which is adverserial fingerprinting & resource constraints
   - Submissions/ - A folder which as CSVs of our various experiments. The csv which was used to generate best result as shown by the public leaderboard is "submission.csv"
3. assignment_2_report.pdf - 2 Pager report of our approaches in this assignment, 


## Assignment 3 Remarks

1. The code files pertaining to the assignment 3 are located in the folder "Assignment_3/tml26_task3"
2. "task_template_trades.py pert" pertains to the experiment we conducted using TRADES
3. "task_template_v1.py" pertains to the code used to train Resnet38/50

## Assignment 4 Remarks

1. The code and report materials for assignment 4 are located in the folder `Assignment_4/`.
2. We evaluated two different black-box watermark forgery approaches. Detailed write-ups are available here:
    - **Method 1: Adaptive Spatial Injection (DIP + JND + Adaptive LPIPS Guardrail)**  
       [`Assignment_4/Adaptive_Spatial_Injection.md`](Assignment_4/Adaptive_Spatial_Injection.md)
    - **Method 2: Averaged Residual Estimation (Classical Denoising + Signal Averaging + Global Alpha)**  
       [`Assignment_4/averaged_residual_details.md`](Assignment_4/averaged_residual_details.md)
3. **Method 1 summary:** Uses Deep Image Prior to extract per-image high-frequency residuals, averages residuals per watermark group, applies JND masking to hide perturbations in textured regions, and enforces an adaptive LPIPS threshold during injection.
4. **Method 2 summary:** Uses Non-Local Means denoising to estimate residuals, averages across the 25-source group to isolate shared watermark signal, then injects with a single globally tuned alpha selected through offline LPIPS sweeps.
5. Our final best leaderboard setting for Assignment 4 came from the simpler averaged-residual pipeline (Method 2), while Method 1 was retained as a more advanced exploratory approach.

