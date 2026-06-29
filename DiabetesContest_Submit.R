#=======================================================================
#
#  IMEN 415 - In-Class Classification Contest
#  과제명: 당뇨망막증(Diabetic Retinopathy) 이진 분류
#  평가지표: F1-Score
#
#  [모델 개요]
#  알고리즘 : Logistic Regression with L1 Regularization (LASSO)
#  라이브러리: glmnet (binomial family)
#  핵심 기법 :
#    1) Feature Engineering
#       - 분포 보정 변환 (log1p, sqrt)
#       - 집계 변수 (sum_ma, max_ma 등)
#       - 중심화 상호작용항 (다중공선성 방지)
#       - 기하학적·임상 변수 상호작용 (v14 신규)
#    2) OOF(Out-of-Fold) F1 직접 최적화
#       - Lambda: AUC 대신 OOF F1로 선택
#       - Threshold: 0.005 단위 정밀 탐색
#    3) Multi-Seed Stratified Ensemble
#       - Stratified 10-Fold CV × 20 Seeds
#       - 20개 모델 예측 확률 평균 → 분산 감소
#
#
#=======================================================================


# ── 라이브러리 ──────────────────────────────────────────────────────────
library(glmnet)   # LASSO / Elastic Net 로지스틱 회귀
library(ggplot2)  # 시각화

# ── 경로 설정 ────────────────────────────────────────────────────────────
datadir <- "/Users/chunsehwan/Library/Mobile Documents/com~apple~CloudDocs/26-1/다변량분석 R/inclass_activity3/imen-415-glm-contest-s-26"


#=======================================================================
#  1. 데이터 로딩
#=======================================================================

DB.train <- read.csv(paste0(datadir, "/db_data_train.csv"), header = TRUE)
DB.test  <- read.csv(paste0(datadir, "/db_data_test.csv"),  header = TRUE)

cat("=== 데이터 현황 ===\n")
cat(sprintf("Train: %d행 × %d열\n", nrow(DB.train), ncol(DB.train)))
cat(sprintf("Test : %d행 × %d열\n", nrow(DB.test),  ncol(DB.test)))
cat("\n[Train 클래스 분포]\n")
print(table(DB.train$Class))
cat(sprintf("양성 비율: %.1f%%\n", 100 * mean(DB.train$Class)))

# ※ 데이터 특이사항:
#   원본 CSV에서 exudate3 컬럼이 중복 기재됨
#   → read.csv가 두 번째 exudate3을 exudate3.1로 자동 변경
#   → exudate4가 아닌 exudate3.1 사용

ma_cols <- paste0("ma", 1:6)
ex_cols <- c("exudate1", "exudate2", "exudate3", "exudate3.1",
             "exudate5", "exudate6", "exudate7", "exudate8")


#=======================================================================
#  2. Feature Engineering
#
#  원칙: 모든 centering 기준값은 Train 데이터에서만 계산하여
#        Test에 동일 적용 (Data Leakage 방지)
#=======================================================================

# --- 2-1. Train 기준 centering 값 계산 ----------------------------------
means <- list()
means$ma_cols     <- setNames(
  lapply(ma_cols, function(col) mean(DB.train[[col]])), ma_cols)
means$sum_ma      <- mean(rowSums(DB.train[, ma_cols]))
means$max_ma      <- mean(apply(DB.train[, ma_cols], 1, max))
means$log_ex1     <- mean(log1p(DB.train$exudate1))
means$log_ex2     <- mean(log1p(DB.train$exudate2))
means$log_sum_ex  <- mean(log1p(rowSums(DB.train[, ex_cols])))
means$log_macdist <- mean(log1p(DB.train$macula_opticdisc_distance))
means$log_opdisc  <- mean(log1p(DB.train$opticdisc_diameter))
means$amfm        <- mean(DB.train$am_fm_classification)


# --- 2-2. Feature 생성 함수 ---------------------------------------------
#
#  [Feature 구성]
#  그룹 A: 분포 보정 변환
#    - exudate1,2: 우편포 (skew > 1.5) → log1p 변환
#    - exudate3  : 우편포 (skew > 3.0) → sqrt 변환
#    - ma1~6     : log1p 변환
#
#  그룹 B: 집계 변수
#    - sum_ma   : 전체 미세혈관류(microaneurysm) 총합
#    - max_ma   : 가장 심한 구역의 미세혈관류 수
#    - ma_nonzero : 병변이 존재하는 구역 수
#    - log_sum_ex : 전체 삼출물(exudate) 총합의 로그
#    - ex_nonzero : 삼출물이 관측된 구역 수
#
#  그룹 C: 중심화 상호작용항 (Centered Interaction)
#    - 공식: (X - mean_train(X)) × (Z - mean_train(Z))
#    - 중심화 이유: Raw 상호작용 ma × log_ex는 ma와의
#      상관관계가 0.93에 달해 다중공선성 발생
#      → 중심화 후 상관관계 0.07로 감소 → 안정적 계수 추정
#    - ma1~6 각각 × log_ex1, log_ex2 = 12개
#    - sum_ma × log_ex1, log_ex2, log_sum_ex = 3개
#    - max_ma × log_ex1, log_ex2 = 2개
#
#  그룹 D: 기하학적·임상 변수 (v14 신규)
#    - macula_opticdisc_distance: 황반-시신경 거리
#    - opticdisc_diameter       : 시신경 직경
#    - 두 변수 및 비율 log 변환
#    - am_fm_classification(AM/FM 알고리즘 결과) × 임상 변수
#    - quality, pre_screening × 임상 변수
#
fe <- function(df, means) {

  ma_cols <- paste0("ma", 1:6)
  ex_cols <- c("exudate1", "exudate2", "exudate3", "exudate3.1",
               "exudate5", "exudate6", "exudate7", "exudate8")

  # 그룹 A: 분포 보정
  df$log_ex1   <- log1p(df$exudate1)
  df$log_ex2   <- log1p(df$exudate2)
  df$sqrt_ex3  <- sqrt(df$exudate3)
  for (col in ma_cols) {
    df[[paste0("log_", col)]] <- log1p(df[[col]])
  }

  # 그룹 B: 집계 변수
  df$sum_ma     <- rowSums(df[, ma_cols])
  df$max_ma     <- apply(df[, ma_cols], 1, max)
  df$ma_nonzero <- rowSums(df[, ma_cols] > 0)
  df$log_sum_ex <- log1p(rowSums(df[, ex_cols]))
  df$ex_nonzero <- rowSums(df[, ex_cols] > 0)

  # 그룹 C: 중심화 상호작용항
  for (m in 1:6) {
    ma_name <- paste0("ma", m)
    ma_mean <- means$ma_cols[[ma_name]]
    df[[paste0("c_", ma_name, "_x_logex1")]] <-
      (df[[ma_name]] - ma_mean) * (df$log_ex1 - means$log_ex1)
    df[[paste0("c_", ma_name, "_x_logex2")]] <-
      (df[[ma_name]] - ma_mean) * (df$log_ex2 - means$log_ex2)
  }
  df$c_summa_x_logex1   <- (df$sum_ma - means$sum_ma) * (df$log_ex1    - means$log_ex1)
  df$c_summa_x_logex2   <- (df$sum_ma - means$sum_ma) * (df$log_ex2    - means$log_ex2)
  df$c_maxma_x_logex1   <- (df$max_ma - means$max_ma) * (df$log_ex1    - means$log_ex1)
  df$c_maxma_x_logex2   <- (df$max_ma - means$max_ma) * (df$log_ex2    - means$log_ex2)
  df$c_summa_x_logsumex <- (df$sum_ma - means$sum_ma) * (df$log_sum_ex - means$log_sum_ex)

  # 그룹 D: 기하학적·임상 변수 (v14 신규)
  df$log_macdist        <- log1p(df$macula_opticdisc_distance)
  df$log_opdisc         <- log1p(df$opticdisc_diameter)
  df$macdist_ratio      <- df$macula_opticdisc_distance / (df$opticdisc_diameter + 1e-6)
  df$log_macdist_ratio  <- log1p(df$macdist_ratio)

  amfm <- df$am_fm_classification
  df$amfm_x_summa   <- (amfm - means$amfm) * (df$sum_ma      - means$sum_ma)
  df$amfm_x_maxma   <- (amfm - means$amfm) * (df$max_ma      - means$max_ma)
  df$amfm_x_logex1  <- (amfm - means$amfm) * (df$log_ex1     - means$log_ex1)
  df$amfm_x_logex2  <- (amfm - means$amfm) * (df$log_ex2     - means$log_ex2)
  df$amfm_x_logmacd <- (amfm - means$amfm) * (df$log_macdist - means$log_macdist)

  df$qual_x_summa  <- df$quality       * df$sum_ma
  df$qual_x_logex1 <- df$quality       * df$log_ex1
  df$qual_x_maxma  <- df$quality       * df$max_ma
  df$pres_x_summa  <- df$pre_screening * df$sum_ma
  df$pres_x_logex1 <- df$pre_screening * df$log_ex1

  df$macd_x_summa  <- (df$log_macdist - means$log_macdist) * (df$sum_ma  - means$sum_ma)
  df$macd_x_logex1 <- (df$log_macdist - means$log_macdist) * (df$log_ex1 - means$log_ex1)

  return(df)
}

DB.train <- fe(DB.train, means)
DB.test  <- fe(DB.test,  means)

cat(sprintf("\n전처리 완료: 원본 %d개 → 파생 포함 %d개 변수\n",
            19, ncol(DB.train) - 1))


#=======================================================================
#  3. 평가 함수
#=======================================================================

# F1-Score: 본 대회의 공식 평가 지표
# F1 = 2 × Precision × Recall / (Precision + Recall)
f1_score <- function(actual, predicted) {
  tp <- sum(actual == 1 & predicted == 1)
  fp <- sum(actual == 0 & predicted == 1)
  fn <- sum(actual == 1 & predicted == 0)
  precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
  recall    <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
  f1        <- ifelse((precision + recall) == 0, 0,
                      2 * precision * recall / (precision + recall))
  return(list(f1 = f1, precision = precision, recall = recall))
}


#=======================================================================
#  4. 모델 학습: Multi-Seed Stratified Ensemble
#
#  [학습 절차 (seed 1개 기준)]
#  Step 1. Stratified 10-Fold 분할
#          양성/음성 샘플 각각 분리하여 fold 배정
#          → 모든 fold에서 클래스 비율 균일 유지
#
#  Step 2. 전체 lambda path 생성
#          glmnet으로 60개의 lambda 후보 자동 생성
#
#  Step 3. OOF(Out-of-Fold) 예측 행렬 구성
#          각 fold를 validation으로 두고 나머지 9개 fold로 학습
#          → 771개 샘플 전체에 대한 OOF 예측 확률 확보
#
#  Step 4. Lambda × Threshold 전수 탐색
#          60 lambdas × 197 thresholds (0.01~0.99, step=0.005)
#          각 조합의 OOF F1 계산 → 최고 조합 선택
#          (cv.glmnet의 AUC 기준이 아닌 F1 직접 최적화)
#
#  Step 5. 최적 lambda로 전체 Train 재학습 → Test 예측
#
#  [앙상블]
#  위 절차를 20개 seed로 반복 → seed마다 fold 배정이 달라져
#  서로 다른 모델 생성 → 20개 Test 예측 확률 평균
#  → 분산 감소 및 일반화 성능 향상
#=======================================================================

cat("\n=== 모델 학습 시작 ===\n")
cat(sprintf("설정: alpha=1.0 (LASSO), K=10, Seeds=20개\n\n"))

# 설정값
K          <- 10
alpha_L1   <- 1.0                          # LASSO 고정
thresh_seq <- seq(0.01, 0.99, by = 0.005) # threshold 후보 197개
seeds      <- c(42, 1, 7, 123, 2024,
                99, 500, 777, 314, 2023,
                11, 55, 888, 1234, 9999,
                17, 256, 1024, 3141, 271828)

feat_cols <- setdiff(names(DB.train), "Class")
x_train   <- as.matrix(DB.train[, feat_cols])
y_train   <- DB.train$Class
x_test    <- as.matrix(DB.test[, feat_cols])

# 클래스 인덱스 (Stratified fold 생성용)
idx_pos <- which(y_train == 1)
idx_neg <- which(y_train == 0)

# 결과 저장
oof_preds  <- matrix(NA, nrow = length(y_train), ncol = length(seeds))
test_preds <- matrix(NA, nrow = nrow(x_test),    ncol = length(seeds))
seed_log   <- data.frame(Seed=integer(), Lambda=double(),
                         Threshold=double(), OOF_F1=double())

# --- Seed 반복 학습 -------------------------------------------------------
for (si in seq_along(seeds)) {
  s <- seeds[si]
  set.seed(s)

  # Step 1: Stratified 10-Fold 분할
  folds <- integer(length(y_train))
  folds[idx_pos] <- sample(rep(1:K, length.out = length(idx_pos)))
  folds[idx_neg] <- sample(rep(1:K, length.out = length(idx_neg)))

  # Step 2: Lambda path 생성 (전체 Train 기준)
  fit_ref    <- glmnet(x_train, y_train, family = "binomial",
                       alpha = alpha_L1, nlambda = 60)
  lambda_seq <- fit_ref$lambda

  # Step 3: OOF 예측 행렬 (n_train × n_lambda)
  oof_mat <- matrix(NA, nrow = length(y_train), ncol = length(lambda_seq))
  for (k in 1:K) {
    idx_val  <- which(folds == k)
    idx_tr   <- which(folds != k)
    fit_k    <- glmnet(x_train[idx_tr, ], y_train[idx_tr],
                       family = "binomial", alpha = alpha_L1,
                       lambda = lambda_seq)
    pred_mat <- predict(fit_k, x_train[idx_val, ], type = "response")
    n_lam    <- min(ncol(pred_mat), ncol(oof_mat))
    oof_mat[idx_val, 1:n_lam] <- pred_mat[, 1:n_lam]
  }

  # Step 4: Lambda × Threshold 전수 탐색 (OOF F1 직접 최적화)
  best_f1  <- -Inf
  best_lam <- NA
  best_t   <- NA
  best_col <- NULL

  for (li in seq_along(lambda_seq)) {
    oof_col <- oof_mat[, li]
    if (any(is.na(oof_col))) next
    f1_vec  <- sapply(thresh_seq,
                      function(t) f1_score(y_train, as.integer(oof_col > t))$f1)
    idx_max <- which.max(f1_vec)
    if (f1_vec[idx_max] > best_f1) {
      best_f1  <- f1_vec[idx_max]
      best_lam <- lambda_seq[li]
      best_t   <- thresh_seq[idx_max]
      best_col <- oof_col
    }
  }

  # Step 5: 최적 lambda로 전체 Train 재학습 → Test 예측
  fit_final          <- glmnet(x_train, y_train, family = "binomial",
                               alpha = alpha_L1, lambda = best_lam)
  oof_preds[, si]    <- best_col
  test_preds[, si]   <- as.numeric(predict(fit_final, x_test, type = "response"))

  seed_log <- rbind(seed_log, data.frame(
    Seed = s, Lambda = best_lam, Threshold = best_t, OOF_F1 = best_f1))

  cat(sprintf("Seed %6d | Lambda=%.5f | T=%.3f | OOF F1=%.4f\n",
              s, best_lam, best_t, best_f1))
}


#=======================================================================
#  5. 앙상블 및 최종 Threshold 결정
#=======================================================================

cat("\n=== 앙상블 결과 ===\n")

# 개별 seed 성능 요약
cat(sprintf("\n[Seed별 OOF F1 요약]\n"))
cat(sprintf("  평균: %.4f | 표준편차: %.4f | 최솟값: %.4f | 최댓값: %.4f\n",
            mean(seed_log$OOF_F1), sd(seed_log$OOF_F1),
            min(seed_log$OOF_F1), max(seed_log$OOF_F1)))

# (1) 단순 평균 앙상블
avg_oof  <- rowMeans(oof_preds)
avg_test <- rowMeans(test_preds)
f1_avg   <- sapply(thresh_seq,
                   function(t) f1_score(y_train, as.integer(avg_oof > t))$f1)
T_avg    <- thresh_seq[which.max(f1_avg)]
F1_avg   <- max(f1_avg)
cat(sprintf("\n단순 평균 앙상블  | OOF F1=%.4f | T=%.3f\n", F1_avg, T_avg))

# (2) 가중 평균 앙상블 (OOF F1 비례 가중치)
weights  <- seed_log$OOF_F1 / sum(seed_log$OOF_F1)
wgt_oof  <- as.numeric(oof_preds  %*% weights)
wgt_test <- as.numeric(test_preds %*% weights)
f1_wgt   <- sapply(thresh_seq,
                   function(t) f1_score(y_train, as.integer(wgt_oof > t))$f1)
T_wgt    <- thresh_seq[which.max(f1_wgt)]
F1_wgt   <- max(f1_wgt)
cat(sprintf("가중 평균 앙상블  | OOF F1=%.4f | T=%.3f\n", F1_wgt, T_wgt))

# (3) 상위 10 seeds 가중 앙상블
top10     <- order(seed_log$OOF_F1, decreasing = TRUE)[1:10]
w10       <- seed_log$OOF_F1[top10] / sum(seed_log$OOF_F1[top10])
top10_oof  <- as.numeric(oof_preds[, top10]  %*% w10)
top10_test <- as.numeric(test_preds[, top10] %*% w10)
f1_top10  <- sapply(thresh_seq,
                    function(t) f1_score(y_train, as.integer(top10_oof > t))$f1)
T_top10   <- thresh_seq[which.max(f1_top10)]
F1_top10  <- max(f1_top10)
cat(sprintf("상위 10 Seeds     | OOF F1=%.4f | T=%.3f\n", F1_top10, T_top10))

# 최고 방법 선택
candidates <- data.frame(
  method = c("단순평균", "가중평균", "상위10"),
  oof_f1 = c(F1_avg, F1_wgt, F1_top10),
  best_T = c(T_avg, T_wgt, T_top10),
  test_prob = I(list(avg_test, wgt_test, top10_test))
)
best_row   <- candidates[which.max(candidates$oof_f1), ]
final_prob <- best_row$test_prob[[1]]
final_T    <- best_row$best_T
final_F1   <- best_row$oof_f1

cat(sprintf("\n→ 채택: %s (OOF F1=%.4f, T=%.3f)\n",
            best_row$method, final_F1, final_T))


#=======================================================================
#  6. 과적합 진단
#=======================================================================

cat("\n=== 과적합 진단 ===\n")

train_pred   <- as.integer(rowMeans(oof_preds) > final_T)  # OOF 기준
result_train <- f1_score(y_train, train_pred)

cat(sprintf("OOF F1    : %.4f\n", final_F1))
cat(sprintf("Precision : %.4f\n", result_train$precision))
cat(sprintf("Recall    : %.4f\n", result_train$recall))
cat("\n[Confusion Matrix (OOF 기준)]\n")
print(table(Actual = y_train, Predicted = train_pred))


#=======================================================================
#  7. 선택된 변수 확인
#=======================================================================

cat("\n=== 선택된 변수 (최고 OOF F1 Seed 기준) ===\n")

best_si      <- which.max(seed_log$OOF_F1)
best_lam_chk <- seed_log$Lambda[best_si]
fit_check    <- glmnet(x_train, y_train, family = "binomial",
                       alpha = alpha_L1, lambda = best_lam_chk)
coef_all     <- as.matrix(coef(fit_check))
coef_nz      <- coef_all[abs(coef_all[, 1]) > 1e-8, , drop = FALSE]
coef_sorted  <- coef_nz[order(abs(coef_nz[, 1]), decreasing = TRUE), , drop = FALSE]

cat(sprintf("전체 변수: %d개 | 선택된 변수: %d개 (L1 zero-shrink 후)\n",
            ncol(x_train), nrow(coef_sorted) - 1))
cat("\n계수 (절댓값 내림차순):\n")
print(round(coef_sorted, 4))


#=======================================================================
#  8. Threshold 탐색 시각화
#=======================================================================

plot_df <- data.frame(Threshold = thresh_seq,
                      F1        = sapply(thresh_seq, function(t)
                        f1_score(y_train, as.integer(rowMeans(oof_preds) > t))$f1))

print(
  ggplot(plot_df, aes(x = Threshold, y = F1)) +
    geom_line(color = "steelblue", linewidth = 1.1) +
    geom_vline(xintercept = final_T, color = "red", linetype = "dashed") +
    annotate("text",
             x     = min(final_T + 0.08, 0.88),
             y     = min(plot_df$F1) + 0.01,
             label = sprintf("T = %.3f\nF1 = %.4f", final_T, final_F1),
             color = "red", size = 3.5) +
    labs(title = "OOF F1-Score vs. Classification Threshold",
         subtitle = "LASSO (alpha=1), 20-Seed Stratified Ensemble",
         x = "Threshold", y = "OOF F1-Score") +
    theme_bw(base_size = 12)
)


#=======================================================================
#  9. Test 예측 및 제출 파일 생성
#=======================================================================

cat("\n=== 제출 파일 생성 ===\n")

final_pred <- as.integer(final_prob > final_T)

submission <- data.frame(
  Id        = seq_len(length(final_pred)),
  Predicted = final_pred
)

cat("[Test 예측 분포]\n")
print(table(Predicted = submission$Predicted))
cat(sprintf("양성 예측 비율: %.1f%%\n",
            100 * mean(submission$Predicted)))

write.csv(submission,
          paste0(datadir, "/sampleSubmission.csv"),
          row.names = FALSE)

cat(sprintf("\n제출 파일 저장 완료: sampleSubmission.csv\n"))
cat(sprintf("최종 설정: 앙상블=%s | Threshold=%.3f | OOF F1=%.4f\n",
            best_row$method, final_T, final_F1))

