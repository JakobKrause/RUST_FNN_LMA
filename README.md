# Reasons for using this framework

I changed to this framework, because i was not happy with the performance of the other one TODO and since I already look for a sparse implementation, I thought this may be a viable starting point. Unfortunately the Levenberg Marquardt Algorithm is not implemented, but I aim to add it.


# 1st Step - Testing

Before I will use this framework, I made sure that the main.rs example works as expected. The execution is fast and the prediction is correct. Now I will try to learn a 1D benchmark function, like in the last framework.

# 2nd Step - Restructuring

I am really unhappy with the structure of the framework and will change it.

Done

[commit: Changes to folder structure](https://github.com/JakobKrause/RUST_FNN_LMA/commit/f8a6b1eca1b6377647f539bffcc3780fe18da8f4)

[commit: Function distributions in modules and error handling](https://github.com/JakobKrause/RUST_FNN_LMA/commit/5c7473459d29ef0f930d8722ec62cb5309aa4f15)

I also change the way the model is build with the help of a "SequentialBuilder", this allows for chainable layer addition and a more compact initalization of the model.

# 3rd Step - Plotting and multimodal benchmark

Non-linear regression did not really work, i suppose the missing regularization is the problem.

# 4rd Step - Regularization

