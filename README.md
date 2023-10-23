## Questionnaire
## Author: Buffett Liu

```bash
## Run this command on terminal
sh run.sh
```


### The steps of the whole project.
1. Doing preprocessing of raw data (merge_result.csv) and temporarily turn Y data into 0/1.

2. Use Elastic Net to get best parameters for preliminary feature selections. (Get 59 features this step)

3. The strategy I use is to randomly select 2 to 10 features from the preliminary feature selections set and train 7 models to see the averaged performance. Ultimately, I store it into mp_res_0910.py.

4. Sequentially select and add the features from the previous top 10 performances until the accuracy doesn't improve. I use 7 models to cross validate the performance. Ultimately, 

5. Finally, I transform the Y data back to original data and use the Zero inflated model to evaluate the features performance.

#### Also, the final result is shown in zip.ipynb.