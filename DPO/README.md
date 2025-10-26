# Direct Preference Optimization (DPO)

DPO came to replace Reinforcement Learning from Human Feedback (RLHF), however to understand DPO we need to understand RLHF.

## AI alignment
[^2] When we train LLMs we pretrain it on massive amounts of data. The model will have some sort of "knowledge" however if we just train the model on the interner it may turn out that the model will make racist remarks - we don't want it, so we want to align the model to what we want from it.

## RL
We can do it with RL(HF), we can create/learn a policy, so a ruleset for the agent (model) to maximize the expected reward. 

In case of language models we can reward the model for selecting a sequence of tokens that we want, e.g. some sort of a helpful non-offensive response. We can reward the model as a simple 0.0-1.0 scoring

|prompt|response|reward(0.0-1.0)|
|------|--------|---------------|
|where is Shanghai?|China|1|
|what is my name?|Heisenberg|0|
|...|...|...|

## DPO
[^1] Great but how de we score those responses as humans, we may not agree as to which response is good or bad. But we may agree when comparing responses e.g.

|prompt|response 1|response 2|chosen as better by human|
|------|----------|----------|-------------------------|
|where is Shanghai?|China|Shanghai does not exist|response 1|
|...|...|...|...|

Amazing, now we have a preference dataset.

We can model our preference as Bradley-Terry model.

$P(y_w > y_l) = \frac{e^{r^*(x, y_w)}}{e^{r^*(x, y_w)} + e^{r^*(x, y_l)}}$

where:

$y_w$ is the winning answer

$y_l$ is the losing answer

We train a ranking model that ranks our responses correctly based on the annotated dataset.

The "DPO preference model" loss:

$L=-\mathbb{E}_{(x, y_w, y_l)\sim D}[log \space \sigma (r_{\varphi}(x, y_w)-r_{\varphi}(x, y_l))]$

To get this equation we can note that:

$\frac{e^A}{e^A+e^B} = \sigma (A-B) \quad ( = \frac{1}{1+e^{-(A-B)}})$

So basically we just maximize the $P$ of getting the "winning" answer. We will train our preference (Bradley-Terry) model to maximize the $P$ of getting the human-prefered answer.

### DPO loss function

$J_{RLHF} = \underset{\pi_{\theta}}{\max}\mathbb{E}_{x \sim D, y \sim \pi_{\theta}(y \vert x)} [r_{\varphi}(x, y) - \beta D_{KL}[\pi_{\theta}(y \vert x) \| \pi_{ref}(y \vert x)]]$

Where:

$r_{\varphi}$ is our reward model

$\pi_{\theta}$ is our LLM we want to train

$\pi_{ref}$ is our reference model

What happens is that we want to maximize the reward of the models response, but we want to stay close to a "regular LLM". So our change to the model does not do "reward hacking" by just outputing garbage. This way we change what the model outputs while staying close to "real language". We can think of it as classic MLE + regualization e.g. MSE + weight decay.

Amazing but this loss is not differentiable, because we sample from the LLMs with some strategies e.g. greedy, ... So we cannot use gradient descent for that.

But there is an analutical solution to this equation:

$\pi_{r}(y \vert x) = \frac{1}{Z(x)}\pi_{ref}(y \vert x) \space exp(\frac{1}{\beta}r(x, y))$

where $Z(x) = \sum_{y}\pi_{ref}(y \vert x) \space exp(\frac{1}{\beta}r(x, y))$

and

$\pi_{r}$ is the optimal policy

ok, amazing but $Z$ is intractable, because we would need to generate all the possible model responses, given a prompt $x$.

Imagine if we had $\pi_{r}$, then we can calculate the reward function as:

$r(x, y) = \beta \space log \frac{\pi_{r}(x \vert y)}{\pi_{ref}(x \vert y)} + \beta \space log \space Z(x)$

We can then recall the Bradley-Terry model

$P(y_w > y_l) = \frac{e^{r^*(x, y_w)}}{e^{r^*(x, y_w)} + e^{r^*(x, y_l)}} = \sigma (r_{\varphi}(x, y_w)-r_{\varphi}(x, y_l))$

then if we apply that model as $r(\cdot, \cdot)$

then we obtain:

$L_{DPO}(\pi_{\theta};\pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)\sim D}[ log \space \sigma ( \beta \space log \frac{\pi_{\theta}(y_w \vert x)}{\pi_{ref}(y_w \vert x)} - \beta \space \frac{\pi_{\theta}(y_l \vert x)}{\pi_{ref}(y_l \vert x)})]$

We can use this loss function to train the LLM to align with our will.

#### How to use this loss

We need to calculate the log probabilities, we do this as:

1. we combine the question and answer to one prompt
$[x \| y_w]$ and $[x \| y_l]$
2. the model outputs the shifted logits e.g.
$[<sos> \| x \| y_w] \space \overset{model}{\rightarrow} [x \| y_w \| ...]$ from that we can choose to have only the log probabilites of $y_w$ taken from the model output, same for $y_l$
3. we sum the log probabilities for both the
$\pi_{\theta}$ and $\pi_{ref}$ for both $y_w$ and $y_l$
4. we calculate the loss and backpropagate

## Disclaimer
> [!CAUTION]
> This repo does not serve to amazingly describe and explain model architectures, it was made to give a broad simplified overview of the models and implement them.

[^1]: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2305.18290

[^2]: Umar J. (2024). Direct Preference Optimization (DPO) explained: Bradley-Terry model, log probabilities, math. https://www.youtube.com/watch?v=hvGa5Mba4c8