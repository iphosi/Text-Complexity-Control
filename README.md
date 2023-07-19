# Text Complexity Control

## Overview
This is the repo for the text complexity control project,
which aims to develop a controllable decoder model that generates German text at different complexity levels.
The repo contains:

- The adapters for multi-complexity-level text generation.
- The source code to train and evaluate the models.
- The evaluation results.

Note: The training data is from a private dataset and thus not provided in this repo.
Actually, any text dataset can be used as an alternative as the training process only requires some short query texts.

## Baseline Model
`ml6team/gpt2-small-german-finetune-oscar` is used as the baseline model.
Please download the model and save it under `./baseline_models/gpt2-german-oscar/CAUSAL_LM`

## Training Strategy
Basically, the baseline model parameters are frozen and the inserted `LoRA` adapter modules are trained on the stylistic
continuation task via `RLHF`.

### Stylistic Continuation
The first n tokens of a text sample will be fed into the model together with one of the following control tokens.
The model should extend the query text at a certain complexity level accordingly.

<table>
    <tr>
        <th>Control Token</th><th>Target Class</th><th>Target Class ID</th>
    </tr>
    <tr>
        <td>[Leichte Sprache]</td><td>Easy Language</td><td>0</td>
    </tr>
    <tr>
        <td>[Einfache Sprache]</td><td>Plain Language</td><td>1</td>
    </tr>
    <tr>
        <td>[Alltagssprache]</td><td>Everyday Language</td><td>2</td>
    </tr>
    <tr>
        <td>[Fachsprache]</td><td>Special Language</td><td>3</td>
    </tr>
</table>

Prompt template: `<ctrl token>: <query text>`

The multi-complexity-level generation task is first treated as four subtasks that correspond to a certain language level respectively.
When training each subtask, only one type of control token will be added in the front of the query text.
During the fusion of adapters, control tokens are randomly sampled.

## Reward Modeling
Reward modeling intends to explicitly inform the model of the output text complexity
so that the model can be optimized in a reinforcement learning manner.

### Regression Reward Function
In regression manner, the response reward is calculated as the difference between the response text simplicity score predicted by a regression model and a predefined baseline:
```math
response\_reward = (simplicity\_score - baseline) \cdot rescaling\_factor \cdot sign(target\_cls\_id)
```
```math
sign(target\_cls\_id) =
\begin{cases}
1& {target\_cls\_id \geq 2}\\
-1& {else}
\end{cases}
```

The optimization process can be regarded as shifting the text complexity distribution away from the baseline (see [Evaluation](#regression)).

Ideally, a regression reward model should predict the human opinion toward the response text complexity.
Due to the scarcity of human feedback data, **F**lesch–**K**incaid **G**rade **L**evel is used as an approximation,
which is highly correlated to the real human feedback (**M**ean **O**pinion **S**core).

<p align="center">
<img src="images/pearson_mos_kincaid.png" alt="Stanford-Alpaca" style="width: 35%; min-width: 300px; display: block; margin: auto;">
</p>

### Classification Reward Function
In classification manner, the response text is classified into four categories i.e. Easy, Plain, Everyday and Special by the reward model.
The predicted logit corresponding to the control token i.e. target class is used as the response reward:
```math
response\_reward = target\_cls\_logit \cdot rescaling\_factor
```

The optimization process can be regarded as sharpening the text complexity distribution (see [Evaluation](#classification)).

`krupper/text-complexity-classification` serves as the classification reward model.

## Adapter Ensemble
Adapters for each complexity level can be trained separately so that the model has a better understanding in each control token.
A drawback of this training strategy is that the adapters should be repeatedly activated or deactivated according to the control demand,
which calls for the adapter ensemble methods.

### Ensemble with Fixed Weights
There exists an in-built method to load several `LoRA` adapters simultaneously with some fixed weights.
```
model.add_weighted_adapter(
    adapters=adapter_names,
    weights=weights,
    adapter_name=ensemble_adapter_name
)
```

By tuning the weight values, a fine-grained control of the output text complexity can be realized.
However, this also leads to a dramatic drop in the model's understanding in control tokens (see [Evaluation](#ensemble-with-fixed-weights)).

### Fusion with Learnable Weights
Currently, there isn't any generalized method to fuse the knowledge of `LoRA` adapters.
But in our case, the subtasks are highly correlated so that the fusion is relatively easy.
The general idea is to load and freeze the task adapters in the attention layer and train an additional fusion adapter inserted in the feed forward layer via the above-mentioned training strategy.

This strategy can to some extent recover the model's understanding in control tokens (see [Evaluation](#fusion-with-learnable-weights)).

## Evaluation
388 query texts are fed into the models to evaluate the output text complexity distribution.
The output texts are then classified by `krupper/text-complexity-classification` and the averaged `FKGL` score is calculated for each class.
The original `.json` files can be found in `./evaluation_peft`.

### Baseline Text Complexity Distribution

<table>
    <tr>
        <th>Model: Baseline</th><th colspan="2">Control Token: None</th>
    </tr>
    <tr>
        <td>Class</td><td>Frequency</td><td>FKGL Mean</td>
    </tr>
    <tr>
        <td>Easy Language</td><td>38</td><td>5.65</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>150</td><td>7.49</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>187</td><td>8.79</td>
    </tr>
    <tr>
        <td>Special Language</td><td>13</td><td>9.90</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>8.02</td>
    </tr>
</table>

The baseline model is most likely to generate plain and everyday language.

### Single Language Style Adaption

#### Regression
The regression reward strategy pushes the text complexity distribution away from the predefined baseline.
This results in the frequency increase of texts at non-target complexity level.
For example, during the plain language adaption, the model also become more likely to generate easy language.

<table>
    <tr>
        <th>Model: Reg-Easy</th><th colspan="2">Control Token: Easy</th><th colspan="2">Control Token: None</th>
    </tr>
    <tr>
        <td>Class</td><td>Frequency</td><td>FKGL Mean</td><td>Frequency</td><td>FKGL Mean</td>
    </tr>
    <tr>
        <td>Easy Language</td><td>65</td><td>3.96</td><td>43</td><td>5.33</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>158</td><td>5.53</td><td>158</td><td>6.94</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>162</td><td>6.82</td><td>179</td><td>7.62</td>
    </tr>
    <tr>
        <td>Special Language</td><td>3</td><td>11.25</td><td>8</td><td>10.67</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>5.85</td><td>388</td><td>7.15</td>
    </tr>
</table>

The easy language adaption model is transferred from the plain language adaption model.
The averaged `FKGL` score of the plain language model is used as the regression baseline.

<table>
    <tr>
        <th>Model: Reg-Plain</th><th colspan="2">Control Token: Plain</th><th colspan="2">Control Token: None</th>
    </tr>
    <tr>
        <td>Class</td><td>Frequency</td><td>FKGL Mean</td><td>Frequency</td><td>FKGL Mean</td>
    </tr>
    <tr>
        <td>Easy Language</td><td>60</td><td>4.67</td><td>27</td><td>4.61</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>169</td><td>6.47</td><td>174</td><td>6.95</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>152</td><td>7.92</td><td>178</td><td>8.19</td>
    </tr>
    <tr>
        <td>Special Language</td><td>7</td><td>11.60</td><td>9</td><td>12.57</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>6.58</td><td>388</td><td>7.49</td>
    </tr>
</table>

The plain language adaption model is trained from scratch.
The averaged `FKGL` score of the baseline model is used as the regression baseline.

<table>
    <tr>
        <th>Model: Reg-Everyday</th><th colspan="2">Control Token: Everyday</th><th colspan="2">Control Token: None</th>
    </tr>
    <tr>
        <td>Class</td><td>Frequency</td><td>FKGL Mean</td><td>Frequency</td><td>FKGL Mean</td>
    </tr>
    <tr>
        <td>Easy Language</td><td>25</td><td>6.41</td><td>24</td><td>5.92</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>145</td><td>8.11</td><td>149</td><td>7.69</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>201</td><td>9.83</td><td>202</td><td>9.16</td>
    </tr>
    <tr>
        <td>Special Language</td><td>17</td><td>10.91</td><td>13</td><td>10.37</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>9.02</td><td>388</td><td>8.44</td>
    </tr>
</table>

The everyday language adaption model is trained from scratch.
The averaged `FKGL` score of the baseline model is used as the regression baseline.

<table>
    <tr>
        <th>Model: Reg-Special</th><th colspan="2">Control Token: Special</th><th colspan="2">Control Token: None</th>
    </tr>
    <tr>
        <td>Class</td><td>Frequency</td><td>FKGL Mean</td><td>Frequency</td><td>FKGL Mean</td>
    </tr>
    <tr>
        <td>Easy Language</td><td>18</td><td>6.16</td><td>20</td><td>6.18</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>130</td><td>7.96</td><td>141</td><td>7.39</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>223</td><td>9.85</td><td>216</td><td>9.25</td>
    </tr>
    <tr>
        <td>Special Language</td><td>17</td><td>10.65</td><td>11</td><td>11.54</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>9.08</td><td>388</td><td>8.48</td>
    </tr>
</table>

The special language adaption model is transferred from the everyday language adaption model.
The averaged `FKGL` score of the everyday language model is used as the regression baseline.

Note: The effectiveness of special language adaption is not significant
since the model requires additional knowledge to generate complex contents
while during the training, the knowledge base of the model remains unchanged.

#### Classification
Compared to regression manner, the classification reward strategy only increase the probability of the target language style.
On the other hand, its influence on the `FKGL` score is much less significant.

Note: Since the baseline distribution is not uniform,
the easy and the special language rewards (logits) can hardly be positive.
In other words, this reward strategy can't be directly applied to easy and special language adaption.

<table>
    <tr>
        <th>Model: Cls-Plain</th><th colspan="2">Control Token: Plain</th><th colspan="2">Control Token: None</th>
    </tr>
    <tr>
        <td>Class</td><td>Frequency</td><td>FKGL Mean</td><td>Frequency</td><td>FKGL Mean</td>
    </tr>
    <tr>
        <td>Easy Language</td><td>33</td><td>5.48</td><td>23</td><td>6.03</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>209</td><td>7.25</td><td>173</td><td>7.54</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>139</td><td>8.66</td><td>185</td><td>8.76</td>
    </tr>
    <tr>
        <td>Special Language</td><td>7</td><td>7.72</td><td>7</td><td>10.63</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>7.61</td><td>388</td><td>8.09</td>
    </tr>
</table>

<table>
    <tr>
        <th>Model: Cls-Everyday</th><th colspan="2">Control Token: Everyday</th><th colspan="2">Control Token: None</th>
    </tr>
    <tr>
        <td>Class</td><td>Frequency</td><td>FKGL Mean</td><td>Frequency</td><td>FKGL Mean</td>
    </tr>
    <tr>
        <td>Easy Language</td><td>35</td><td>6.88</td><td>32</td><td>5.83</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>67</td><td>7.61</td><td>137</td><td>7.50</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>279</td><td>8.54</td><td>210</td><td>9.01</td>
    </tr>
    <tr>
        <td>Special Language</td><td>7</td><td>11.32</td><td>9</td><td>8.70</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>8.28</td><td>388</td><td>8.21</td>
    </tr>
</table>

#### Combination
It is possible to combine the regression and the classification strategies.
The general idea is to first shift the complexity distribution and then sharpen it
so that the overall `FKGL` score gets further improved while the frequency of non-target language style won't increase too much.

<table>
    <tr>
        <th>Model: Reg-Cls-Plain</th><th colspan="2">Control Token: Plain</th><th colspan="2">Control Token: None</th>
    </tr>
    <tr>
        <td>Class</td><td>Frequency</td><td>FKGL Mean</td><td>Frequency</td><td>FKGL Mean</td>
    </tr>
    <tr>
        <td>Easy Language</td><td>39</td><td>5.79</td><td>24</td><td>5.49</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>185</td><td>6.76</td><td>159</td><td>7.30</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>158</td><td>8.31</td><td>199</td><td>8.64</td>
    </tr>
    <tr>
        <td>Special Language</td><td>6</td><td>8.00</td><td>6</td><td>10.05</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>7.31</td><td>388</td><td>7.92</td>
    </tr>
</table>

### Multiple Language Style Adaption
It's non-trivial to train a relatively small-sized model from scratch to recognize several language styles and control tokens at the same time.
A more feasible way to do this would be fusing the knowledge of the separately pretrained adapters.

<table>
    <tr>
        <th colspan="2">Model: Ensemble-Plain-Everyday</th><th colspan="2">Control Token: Plain</th><th colspan="2">Control Token: Everyday</th>
    </tr>
    <tr>
        <td>Weights</td><td>Class</td><td>Frequency</td><td>FKGL Mean</td><td>Frequency</td><td>FKGL Mean</td>
    </tr>
    <tr>
        <td rowspan="5">0.2:0.8</td><td>Easy Language</td><td>19</td><td>7.29</td><td>34</td><td>6.04</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>144</td><td>7.58</td><td>150</td><td>7.60</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>214</td><td>9.07</td><td>195</td><td>9.40</td>
    </tr>
    <tr>
        <td>Special Language</td><td>11</td><td>8.36</td><td>9</td><td>9.46</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>8.41</td><td>388</td><td>8.41</td>
    </tr>
    <tr>
        <td rowspan="5">0.8:0.2</td><td>Easy Language</td><td>40</td><td>4.93</td><td>37</td><td>4.43</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>153</td><td>6.96</td><td>159</td><td>7.24</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>183</td><td>8.13</td><td>182</td><td>8.32</td>
    </tr>
    <tr>
        <td>Special Language</td><td>12</td><td>9.58</td><td>10</td><td>9.73</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>7.38</td><td>388</td><td>7.54</td>
    </tr>
    <tr>
        <td rowspan="5">0.5:0.5</td><td>Easy Language</td><td>36</td><td>6.28</td><td>30</td><td>5.59</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>158</td><td>7.31</td><td>165</td><td>7.23</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>187</td><td>8.30</td><td>183</td><td>8.96</td>
    </tr>
    <tr>
        <td>Special Language</td><td>7</td><td>9.06</td><td>10</td><td>7.22</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>7.72</td><td>388</td><td>7.92</td>
    </tr>
    <tr>
        <td rowspan="5">1.2:1.0</td><td>Easy Language</td><td>67</td><td>4.39</td><td>59</td><td>4.39</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>138</td><td>6.65</td><td>132</td><td>6.66</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>180</td><td>7.70</td><td>190</td><td>8.29</td>
    </tr>
    <tr>
        <td>Special Language</td><td>3</td><td>10.57</td><td>7</td><td>7.45</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>6.78</td><td>388</td><td>7.13</td>
    </tr>
</table>

The `FKGL` score of the output texts can be flexibly manipulated by tuning the weight values.
However, the score also becomes less sensitive to the control tokens.

Note: The grammatical correctness of the output texts drops dramatically if the weight values are too large.

<table>
    <tr>
        <th colspan="2">Model: Fusion-Plain-Everyday</th><th colspan="2">Control Token: Plain</th><th colspan="2">Control Token: Everyday</th>
    </tr>
    <tr>
        <td>Init Weights</td><td>Class</td><td>Frequency</td><td>FKGL Mean</td><td>Frequency</td><td>FKGL Mean</td>
    </tr>
    <tr>
        <td rowspan="5">0.5:0.8</td><td>Easy Language</td><td>45</td><td>5.30</td><td>35</td><td>5.34</td>
    </tr>
    <tr>
        <td>Plain Language</td><td>160</td><td>7.19</td><td>162</td><td>7.35</td>
    </tr>
    <tr>
        <td>Everyday Language</td><td>175</td><td>8.55</td><td>181</td><td>9.53</td>
    </tr>
    <tr>
        <td>Special Language</td><td>8</td><td>10.18</td><td>10</td><td>11.57</td>
    </tr>
    <tr>
        <td>Summary</td><td> 388</td><td>7.65</td><td>388</td><td>8.30</td>
    </tr>
</table>

The learnable weighting strategy better preserves the knowledge of the control tokens.
