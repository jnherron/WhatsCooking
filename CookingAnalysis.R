library(jsonlite) 
library(tidyverse)
library(tidytext)
library(tidymodels)
library(stringr)
library(vroom)
library(textrecipes)
library(dplyr)
library(discrim)
library(embed)
library(keras)
library(reticulate)
library(kernlab)
library(themis)
library(naivebayes)
library(glmnet)
library(kknn)
library(stopwords)



train_data <- read_file("train.json") %>%
  fromJSON()
test_data <- read_file("test.json") %>%
  fromJSON()

train_data <- train_data %>%
  mutate(ingredients=sapply(ingredients, FUN=function(lst){
    paste(lst, collapse=" ")
  }))

test_data <- test_data %>%
  mutate(ingredients = sapply(ingredients, function(lst){
    paste(lst, collapse=" ")
  }))


# TFIDF ------------------------------------------------------------------------

my_recipe <- recipe(cuisine ~ ingredients, data = train_data) %>%
  step_tokenize(ingredients) %>%
  step_tokenfilter(ingredients, max_tokens = 1500) %>%
  step_tfidf(ingredients)

# baked <- bake(prep(my_recipe), new_data = train_data)
# View(baked)


# Random Forest ----------------------------------------------------------------

forest_mod <- rand_forest(mtry = tune(), min_n=tune(), trees=150) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

tuning_grid <- grid_regular(mtry(range=c(1,50)), min_n(), levels=3)
folds <- vfold_cv(train_data, v = 5, repeats=1)

cv_results <- forest_wf %>%
  tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(accuracy))
best_tune <- cv_results %>%
  select_best(metric="accuracy")

final_wf <- forest_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data=train_data)

forest_preds <- predict(final_wf, new_data=test_data, type="class")

kag_sub <- data.frame(id = test_data$id, cuisine = forest_preds$.pred_class)
vroom_write(x=kag_sub, file="./Forest_tokens1000_trees100_l3v5.csv", delim=",")


# SVN Poly ---------------------------------------------------------------------

# svm_poly_mod <- svm_poly(degree=1, cost=.0131) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab", prob.model = TRUE)
# 
# svm_poly_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(svm_poly_mod) %>%
#   fit(data=train_data)
# 
# svm_poly_preds <- predict(svm_poly_wf, new_data=test_data, type="class")
# 
# kag_sub <- data.frame(id = test_data$id, cuisine = svm_poly_preds$.pred_class)
# vroom_write(x=kag_sub, file="./SvmPoly.csv", delim=",")

# SVN Linear -------------------------------------------------------------------

# svm_linear_mod <- svm_linear(cost=.0131) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab", prob.model = TRUE)
# 
# svm_linear_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(svm_linear_mod) %>%
#   fit(data=train_data)
# 
# svm_linear_preds <- predict(svm_linear_wf, new_data=test_data, type="class")
# 
# kag_sub <- data.frame(id = test_data$id, cuisine = svm_linear_preds$.pred_class)
# vroom_write(x=kag_sub, file="./SvmLinear.csv", delim=",")

# Naive Bayes ------------------------------------------------------------------

# nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")
# 
# nb_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(nb_mod)
# 
# tuning_grid <- grid_regular(Laplace(), smoothness(), levels=3)
# folds <- vfold_cv(train_data, v = 5, repeats=1)
# 
# cv_results <- nb_wf %>%
#   tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(roc_auc))
# best_tune <- cv_results %>%
#   select_best(metric="roc_auc")
# 
# final_wf <- nb_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data=train_data)
# 
# nb_preds <- predict(final_wf, new_data=test_data, type="class")
# 
# kag_sub <- data.frame(id = test_data$id, cuisine = nb_preds$.pred_class)
# vroom_write(x=kag_sub, file="./NB_tokens1000.csv", delim=",")


# Boost ------------------------------------------------------------------------

# boost_mod <- boost_tree(tree_depth=tune(), trees=tune(), learn_rate=tune()) %>%
#   set_engine("lightgbm") %>%
#   set_mode("regression")
# 
# boost_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(boost_mod)
# 
# grid_of_tuning_params <- grid_regular(tree_depth(), trees(), learn_rate(), levels=3)
# folds <- vfold_cv(train_data, v=5, repeats=1)
# 
# CV_results <- boost_wf %>%
#   tune_grid(resamples=folds, grid=grid_of_tuning_params, metrics=metric_set(roc_auc))
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# final_wf <- boost_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = train_data)
# 
# boost_preds <- predict(final_wf, new_data=test_data, type="class")
# 
# kag_sub <- data.frame(id = test_data$id, cuisine = boost_preds$.pred_class)
# vroom_write(x=kag_sub, file="./Boost.csv", delim=",")

# Bart -------------------------------------------------------------------------

# bart_mod <- parsnip::bart(trees = tune()) %>%
#   set_engine("dbarts") %>%
#   set_mode("regression")
# 
# bart_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(bart_mod)
# 
# grid_of_tuning_params <- grid_regular(trees(), levels=3)
# folds <- vfold_cv(train_data, v=5, repeats=1)
# 
# CV_results <- bart_wf %>%
#   tune_grid(resamples=folds, grid=grid_of_tuning_params, metrics=metric_set(roc_auc))
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# final_wf <- bart_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = train_data)
# 
# bart_preds <- predict(final_wf, new_data=test_data, type="class")
# 
# kag_sub <- data.frame(id = test_data$id, cuisine = bart_preds$.pred_class)
# vroom_write(x=kag_sub, file="./Bart.csv", delim=",")

# Stacking (5 Models) ----------------------------------------------------------

# h2o::h2o.init()
# 
# auto_mod <- auto_ml() %>%
#   set_engine("h2o", max_models=5) %>%
#   set_mode("regression")
# 
# auto_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(auto_mod) %>%
#   fit(data=train_data)
# 
# auto_preds <- predict(auto_wf, new_data=test_data, type="class")
# 
# kag_sub <- data.frame(id = test_data$id, cuisine = auto_preds$.pred_class)
# vroom_write(x=kag_sub, file="./Stack_5.csv", delim=",")

# Stacking (2 Models) ----------------------------------------------------------

# h2o::h2o.init()
# 
# auto_mod <- auto_ml() %>%
#   set_engine("h2o", max_models=2) %>%
#   set_mode("regression")
# 
# auto_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(auto_mod) %>%
#   fit(data=train_data)
# 
# auto_preds <- predict(auto_wf, new_data=test_data, type="class")
# 
# kag_sub <- data.frame(RefId = test_data$RefId, IsBadBuy = auto_preds$.pred_1)
# vroom_write(x=kag_sub, file="./Stack_2.csv", delim=",")


# KNN --------------------------------------------------------------------------

# knn_mod <- nearest_neighbor(neighbors=tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kknn")
# 
# knn_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(knn_mod)
# 
# tuning_grid <- grid_regular(neighbors(), levels=3)
# folds <- vfold_cv(train_data, v = 5, repeats=1)
# 
# cv_results <- knn_wf %>%
#   tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(roc_auc))
# best_tune <- cv_results %>%
#   select_best(metric="roc_auc")
# 
# final_wf <- knn_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data=train_data)
# 
# knn_preds <- predict(final_wf, new_data=test_data, type="class")
# 
# kag_sub <- data.frame(RefId = test_data$RefId, IsBadBuy = knn_preds$.pred_1)
# vroom_write(x=kag_sub, file="./Knn.csv", delim=",")



