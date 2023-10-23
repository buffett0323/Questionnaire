## Questionnaire
## Author: Buffett Liu

```bash
## Run this command on terminal
sh run.sh
```

### Extract important features that significantly on behalf of the tendency people fill out the questionnaire. 

### The steps of the whole project.
Firstly, I do the preprocessing of the raw data (merge_result.csv) and temporarily turn Y data into a binary classification task.
Secondly, I use Elastic Net to select for preliminary feature selections. After that, I randomly select 2 to 10 features from the preliminary feature selection set and train 7 models to see the averaged performance and store those results.
Furthermore, I sequentially select and add the features from the previous top 10 performances until the accuracy doesn’t improve. 
Finally, I transform the Y data back to the original data and use the Zero inflated model to evaluate the feature performance and get the result of important features whose p values in the model are less than 0.05.


#### Also, the final result is shown in zip.ipynb.
