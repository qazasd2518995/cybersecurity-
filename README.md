# Project: Lightweight Markov Password Strength Demo

## Description
This project showcases a simple machine learning approach for password strength assessment using a character-level Markov model and compares its effectiveness with the popular zxcvbn password strength estimator. It is based on the idea of using a lightweight Markov model to evaluate passwords and predict their attack risk (i.e., how easily an attacker might guess them). The demo trains a Markov chain on a sample password list and then produces various outputs: it scores example passwords with both the Markov model and zxcvbn, highlights weak passwords, examines tricky cases (like keyboard patterns or passphrases), and estimates guess ranking. The goal is to illustrate how a data-driven model can learn password patterns and to understand where it agrees or disagrees with an expert rule-based system like zxcvbn.

## Setup & Installation

**Repository**
Full source code is available on GitHub at `https://github.com/qazasd2518995/cybersecurity-` (as mentioned in the report). You can clone it directly:
```bash
git clone https://github.com/qazasd2518995/cybersecurity-.git
cd cybersecurity-
```

**Prerequisites:**
*   Python 3.6+
*   Required Python libraries:
    *   `zxcvbn`
    *   `numpy` (for AUC calculation)

**Installation:**
1.  Clone or download this repository.
2.  Ensure Python is installed.
3.  Install the required libraries:
    ```bash
    pip install zxcvbn numpy
    ```
4.  (Optional) It's recommended to use a Python virtual environment.

## Files in this project:
*   `markov_demo.py`: The main Python script containing the Markov model implementation and the demo routine.
*   `sample_passwords.txt`: A small sample dataset of passwords (one per line) used for training the Markov model in the demo. Create this file with a few passwords for testing, e.g.:
    ```
    123456
    password
    qwerty
    admin
    P@$$wOrd
    apple
    secret123
    ThisIsALongPassword12345
    ```
*   `README.md`: Project documentation and usage instructions (this file).

## Usage

**Running the Demo:**
The demo is contained in `markov_demo.py` and can be run directly with Python.
Ensure `sample_passwords.txt` is present in the same directory as `markov_demo.py`, or update the `sample_password_file` variable at the bottom of `markov_demo.py` to point to the correct location on your system.

Once the `sample_passwords.txt` file is prepared and the path is correctly set (if modified), run the script:
```bash
python markov_demo.py
```
This will execute the `run_demo()` function, producing a series of printed outputs to the console.

## Example (CLI) Output (abridged):
(The actual output will vary based on the `sample_passwords.txt` content and randomness in Python's set orders if not handled, but the structure will be similar to the report's example)

## Project Structure and Code Overview
The core of the project is the `MarkovPasswordModel` class in `markov_demo.py`. Its functionality includes:
*   `train(passwords_list)`: trains the n-gram model on a list of passwords.
*   `get_log_likelihood(password)`: computes the log-likelihood score of a given password under the model.

Helper functions include:
*   `get_zxcvbn_details(password)`: for zxcvbn scores and weak/strong labels.
*   Functions for precision/recall/F1 and ROC/PR curve data generation (e.g., `calculate_classification_metrics`, `get_roc_pr_data`).
*   `approximate_guess_rank(...)`: for demonstrating guess ranking.
*   `run_policy_experiments(model, zxcvbn_func)`: for testing password variations.
*   `run_demo(password_file_path)`: which orchestrates the entire process.

The code is commented, and parameters like `n` for the Markov model or test passwords can be adjusted within `markov_demo.py`.

## Results: Key Findings
(Based on the provided report)
*   **Markov model vs zxcvbn (overall)**: The Markov model's weakness scores largely agree with zxcvbn on very weak (e.g., "123456", "password") and very strong (long random strings) passwords. However, they sometimes disagree on edge cases.
*   The Markov model, being purely frequency-based, can mistakenly view some patterned passwords (e.g., "zzzzzzzz" if underrepresented in small training set) as strong if those patterns weren't common in training data, whereas zxcvbn correctly labels such known patterns as weak.
*   The Markov model might label a password as very weak if it appeared in the training data (memorization).
*   Zxcvbn has built-in knowledge of sequences, years, dates, etc., which the Markov model will only catch if those exact sequences were frequent in data.
*   **Metrics**: Illustrative Precision, Recall, F1-score, and AUC values demonstrate the model's capability to distinguish weak passwords, with performance expected to improve with larger training sets.
*   **Guess ranking**: Demonstrates how the Markov model orders password guessing, providing a way to communicate strength.
*   **Policy impacts**:
    *   Increasing length (with unpredictable content) significantly improves strength for both models.
    *   Adding symbols/case in predictable ways offers limited improvement.
    *   The Markov model's sensitivity depends on its training data.

## Limitations
(Based on the provided report)
*   **Small Training Data**: The demo uses a small sample of passwords, insufficient for generalization. Results are illustrative.
*   **Not a Comprehensive Meter**: The Markov model alone doesn't check user-specific weak choices or provide detailed feedback like zxcvbn.
*   **Deterministic vs. Adaptive**: The demo doesn't show online learning for the Markov model.
*   **No GUI or Visualization**: Console-based output. ROC/PR points are printed for external plotting.
*   **Security of using a model**: Deploying models with embedded leaked data client-side has security implications.
*   **Python performance**: Python is for demonstration; production systems might need optimized implementations.

## Screenshots / Visuals
A plot of the ROC curve and Precision-Recall curve can be derived from the demo's output data points. These curves illustrate the Markov model's ability to trade off between catching weak passwords and avoiding false alarms. With more data, these curves would be smoother and likely show better performance.
*(The script `markov_demo.py` will print the data points for these curves; actual plotting would require a library like Matplotlib.)*

## Key Findings Recap
*   A 4-gram Markov model can quantitatively measure password "strength" based on likelihood.
*   Comparison with zxcvbn shows agreement on extremes but divergence on certain patterns, highlighting complementary strengths.
*   A hybrid approach combining data-driven and rule-based methods is promising.
*   Guess ranking offers an intuitive way to predict attack risk.

## Limitations & Next Steps
The current implementation is a proof of concept. For extensions:
*   Train on a larger dataset (e.g., a RockYou subset).
*   Experiment with different `n-gram` sizes and smoothing methods.
*   Implement and compare with neural network approaches (e.g., RNNs).
*   Develop a hybrid meter combining Markov/ML outputs with zxcvbn-like checks.
*   Create a user interface for better visualization and interaction.
*   Consider security implications for deployment.
*   Validate predictions against real-world cracking tool performance.

## Conclusion (of README)
This project provides a hands-on illustration of using a Markov model for password strength assessment, underscoring the value of data-driven approaches while showing their limitations and the importance of combining them with expert knowledge. It serves as a learning tool for understanding password security from an ML perspective.

## Team Contribution
(As per the report)
| 組員（學號）       | 主要工作內容                                               | 備註             |
| ------------------ | ---------------------------------------------------------- | ---------------- |
| 徐芳澄（B11130108） | 文獻蒐集與撰寫，簡報設計                                     | 負責APA 引用格式 |
| 鄭兆翔（A11317005） | Demo 開發（4-gram Markov），實驗與結果撰寫，整合全文報告       | 第一作者         |
| 莊又翰（B11102021） | 資料清理，ROC/AUC 腳本，圖表製作，校對排版                     | 交叉檢查數值     |

## References
(As per the report)
[1] National Institute of Standards and Technology (NIST). Electronic Authentication Guideline. NIST Special Publication 800-63, April 2006.
[2] Paul A. Grassi, Michael E. Garcia, and Joan M. Fenton. Digital Identity Guidelines: Authentication and Lifecycle Management. NIST Special Publication 800-63B, June 2017.
[3] Arvind Narayanan and Vitaly Shmatikov. "Fast Dictionary Attacks on Passwords Using Time–Space Tradeoff." In Proceedings of the 12th ACM Conference on Computer and Communications Security (CCS 2005), pp. 364–372, 2005.
[4] Matt Weir, Sudhir Aggarwal, Michael Collins, and Henry Stern. "Password Cracking Using Probabilistic Context-Free Grammars." In Proceedings of the 30th IEEE Symposium on Security and Privacy (S&P 2009), pp. 391–405, 2009.
[5] William Melicher, Blase Ur, Sean M. Segreti, Saranga Komanduri, Lujo Bauer, Nicolas Christin, and Lorrie F. Cranor. "Fast, Lean, and Accurate: Modeling Password Guessability Using Neural Networks." In 25th USENIX Security Symposium (USENIX Security 16), pp. 175–191, 2016.
[6] Briland Hitaj, Paolo Gasti, Giuseppe Ateniese, and Fernando Pérez-Cruz. "PassGAN: A Deep Learning Approach for Password Guessing." In Proceedings of the 2019 IEEE International Conference on Information Security and Privacy Protection, pp. 146–157, 2019.
[7] Daniel V. Wheeler. "zxcvbn: Low-Budget Password Strength Estimation." Dropbox Inc., Technical Report, 2016. `https://github.com/dropbox/zxcvbn`
[8] Imperva Inc. "The RockYou Password Leak: Analysis of 32 Million Compromised Credentials." Technical Report, December 2009.
