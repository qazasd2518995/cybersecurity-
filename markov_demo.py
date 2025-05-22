import collections
import math
from zxcvbn import zxcvbn
import numpy as np # For threshold generation

# --- MarkovModel Class (from previous version, adapted for n=4 and minor improvements) ---
class MarkovPasswordModel:
    def __init__(self, n=4): # Changed to n=4 as per paper
        self.n = n
        self.transitions = collections.defaultdict(collections.Counter)
        self.char_counts = collections.Counter() # Used for vocab size in smoothing
        self.vocab = set()

    def train(self, passwords):
        if not passwords:
            print("Warning: No passwords provided for training.")
            return

        for password in passwords:
            if not password:
                continue
            
            # Add all unique characters from passwords to vocab
            for char in password:
                self.vocab.add(char)

            padded_password = '^' * (self.n - 1) + password + '$' # Start/end markers

            for i in range(len(padded_password) - self.n + 1):
                gram = padded_password[i : i + self.n]
                prefix = gram[:-1]
                next_char = gram[-1]
                
                self.transitions[prefix][next_char] += 1
                self.char_counts[next_char] += 1 # Count all chars for smoothing
        
        # Add special tokens to vocab if they were used in padding
        self.vocab.add('^')
        self.vocab.add('$')
        
        if not self.vocab:
            print("Warning: Vocabulary is empty after training. Model might not work correctly.")


    def get_log_likelihood(self, password, smoothing_k=1):
        if not self.vocab: # Check if model was trained / vocab exists
            # print("Warning: Model vocabulary is empty. Returning -inf log-likelihood.")
            return -float('inf')

        if not password:
            return -float('inf')

        log_likelihood = 0.0
        padded_password = '^' * (self.n - 1) + password + '$'

        for i in range(len(padded_password) - self.n + 1):
            gram = padded_password[i : i + self.n]
            prefix = gram[:-1]
            next_char = gram[-1]

            prefix_total_transitions = sum(self.transitions[prefix].values())
            char_transition_count = self.transitions[prefix][next_char]
            
            # Laplace smoothing
            prob = (char_transition_count + smoothing_k) / (prefix_total_transitions + smoothing_k * len(self.vocab))
            
            if prob > 0:
                log_likelihood += math.log(prob)
            else:
                log_likelihood += -float('inf') 
        return log_likelihood

# --- New functions for metrics and experiments ---

def get_zxcvbn_labels(passwords_data):
    """ Get 'weak' (True) or 'strong' (False) labels from zxcvbn """
    labels = []
    scores = []
    for pwd_data in passwords_data:
        # Assuming pwd_data can be a simple string or a dict with 'password' key
        pwd = pwd_data if isinstance(pwd_data, str) else pwd_data['password']
        z_score = zxcvbn(pwd)['score']
        scores.append(z_score)
        labels.append(z_score < 3) # True if weak (score 0, 1, 2)
    return np.array(labels), np.array(scores)

def calculate_prf1(y_true, y_pred_scores, threshold):
    """ Calculates Precision, Recall, F1 for Markov model predictions """
    # Markov: higher log-likelihood (less negative) is weaker.
    # So, if score > threshold, predict as weak.
    y_pred_binary = y_pred_scores > threshold 

    tp = np.sum((y_pred_binary == True) & (y_true == True))
    fp = np.sum((y_pred_binary == True) & (y_true == False))
    fn = np.sum((y_pred_binary == False) & (y_true == True))
    tn = np.sum((y_pred_binary == False) & (y_true == False))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # Same as TPR
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    tpr = recall # True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Positive Rate
    
    return precision, recall, f1, tpr, fpr

def generate_roc_pr_data(y_true, markov_scores_for_roc):
    """ Generates data points for ROC and PR curves """
    # Markov scores are log-likelihoods; more negative = stronger.
    # For ROC/PR, we need to vary a threshold.
    # Let's test thresholds across the range of observed Markov scores.
    # Lower scores (more negative) should correspond to "strong" predictions.
    # Higher scores (less negative) should correspond to "weak" predictions.
    
    # Sort scores to define thresholds: unique sorted log-likelihoods
    # Thresholds will be applied such that if markov_score > threshold, it's predicted weak.
    thresholds = sorted(list(set(markov_scores_for_roc))) 
    if not thresholds: return [], []

    # Add a value slightly above max and below min to ensure full curve range
    if thresholds:
        thresholds = np.concatenate(([thresholds[0] - 1], thresholds, [thresholds[-1] + 1]))
    else: # only one unique score or empty
        thresholds = np.array([np.min(markov_scores_for_roc) -1, np.max(markov_scores_for_roc) + 1])


    roc_points = [] # (fpr, tpr)
    pr_points = []  # (recall, precision)

    for thresh in thresholds:
        # Predict weak if markov_score > thresh
        # (because higher log-likelihood means weaker according to model)
        y_pred_roc = markov_scores_for_roc > thresh
        
        tp = np.sum((y_pred_roc == True) & (y_true == True))
        fp = np.sum((y_pred_roc == True) & (y_true == False))
        fn = np.sum((y_pred_roc == False) & (y_true == True))
        tn = np.sum((y_pred_roc == False) & (y_true == False))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # Handle division by zero
        recall = tpr

        roc_points.append((fpr, tpr))
        # For PR curve, typically precision is on y-axis, recall on x-axis
        # Only add point if precision or recall is non-zero to avoid cluttering (0,0)
        if recall > 0 or precision > 0:
             pr_points.append((recall, precision))

    # Sort for plotting: ROC by FPR, PR by Recall
    roc_points.sort(key=lambda x: x[0])
    pr_points.sort(key=lambda x: x[0])
    
    # Ensure (0,0) and (1,1) for ROC, and potentially (0, value) and (1, value) for PR
    if not any(p[0] == 0 and p[1] == 0 for p in roc_points):
        roc_points.insert(0, (0.0, 0.0))
    if not any(p[0] == 1 and p[1] == 1 for p in roc_points):
        roc_points.append((1.0, 1.0))

    # Remove duplicate points for cleaner plotting data
    roc_points = sorted(list(set(roc_points)))
    pr_points = sorted(list(set(pr_points)))
    if not pr_points or pr_points[0] != (0.0, pr_points[0][1] if pr_points else 1.0): # Start PR curve from recall=0
        # Find precision at recall=0 (usually 1.0 if any positives exist, or based on first point)
        # This is a bit heuristic for PR curve start.
        # A common convention is to extend to (0, p_at_first_recall_point)
        pass # Plotting libraries usually handle this.

    return roc_points, pr_points


def approximate_guess_rank(markov_model, training_passwords, test_passwords_list):
    """ Approximates guess rank for test passwords """
    print("\n--- Approximate Guess Rank ---")
    print("Note: Based on sorting all training + test passwords by Markov log-likelihood.")
    print("Lower rank (closer to 1) means model finds it weaker/more guessable.\n")

    all_passwords_for_ranking = list(set(training_passwords + test_passwords_list))
    
    scored_passwords = []
    for pwd in all_passwords_for_ranking:
        score = markov_model.get_log_likelihood(pwd)
        scored_passwords.append({'password': pwd, 'markov_score': score})
    
    # Sort by Markov score (ascending log-likelihood means weaker is ranked higher, so we sort descending for rank)
    # No, ascending log-likelihood (less negative) means weaker. So sort ascending.
    # Rank 1 is the weakest.
    scored_passwords.sort(key=lambda x: x['markov_score'], reverse=False) # Weakest first (higher log-likelihood)

    ranks = {}
    for i, item in enumerate(scored_passwords):
        if item['password'] in test_passwords_list:
            ranks[item['password']] = {
                'rank': i + 1, 
                'markov_score': item['markov_score'],
                'total_evaluated_for_rank': len(scored_passwords)
            }
            
    print(f"{'Test Password':<30} | {'Approx. Guess Rank':<20} | {'Markov LogLikelihood':<20}")
    print("-" * 70)
    for pwd in test_passwords_list:
        if pwd in ranks:
            print(f"{pwd:<30} | {ranks[pwd]['rank']:<20} | {ranks[pwd]['markov_score']:<20.4f}")
        else:
            print(f"{pwd:<30} | {'Not found in ranking set':<20} | {'N/A':<20}")
    print(f"Total passwords considered for ranking: {len(scored_passwords)}")
    return ranks


def run_policy_experiments(markov_model):
    """ Runs simple password policy experiments """
    print("\n--- Password Policy Experiments ---")
    
    policies = {
        "Length Increase": [
            ("base", "apple"),
            ("base+1", "apple1"),
            ("base+2", "apple12"),
            ("base+3", "apple123"),
            ("base+long", "appleorangebanana"),
        ],
        "Symbol Inclusion / Substitution": [
            ("plain", "password"),
            ("capitalized", "Password"),
            ("leet_simple", "P@ssword"),
            ("leet_complex", "P@$$wOrd"),
            ("with_symbols", "password!@#"),
        ]
    }

    print(f"{'Policy':<25} | {'Variation':<20} | {'Password':<25} | {'Markov LogLik (n=4)':<20} | {'zxcvbn Score':<15}")
    print("-" * 110)

    for policy_name, variations in policies.items():
        for variation_name, pwd_to_test in variations:
            m_score = markov_model.get_log_likelihood(pwd_to_test)
            z_results = zxcvbn(pwd_to_test)
            z_score = z_results['score']
            print(f"{policy_name:<25} | {variation_name:<20} | {pwd_to_test:<25} | {m_score:<20.4f} | {z_score:<15}")
        print("-" * 110)

# --- Main Demo Function ---
def run_demo(password_file_path):
    passwords_for_training = []
    try:
        with open(password_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                passwords_for_training.append(line.strip())
    except FileNotFoundError:
        print(f"Error: Training password file not found at {password_file_path}")
        return

    print("="*70)
    print("Machine Learning for Password Strength: Markov Model Demo (n=4)")
    print("="*70)
    print("IMPORTANT NOTE:")
    print("This demo uses a VERY SMALL training dataset for illustrative purposes.")
    print(f"({len(passwords_for_training)} passwords from {password_file_path}).")
    print("Results (AUC, P/R/F1, ranks) are illustrative for this small dataset and WILL NOT match")
    print("claims made for models trained on large datasets (e.g., 70k RockYou samples).")
    print("The primary purpose is to demonstrate the *methodology*.")
    print("="*70 + "\n")

    # Initialize and train the Markov model (n=4)
    markov_model = MarkovPasswordModel(n=4)
    markov_model.train(passwords_for_training)
    print(f"Model training complete. Vocabulary size: {len(markov_model.vocab)}\n")

    # Test passwords for general evaluation
    test_passwords_eval = [
        "123456", "password", "qwerty", "admin",
        "P@$$wOrd", "Str0ngP@sswOrd!", "secret123",
        "dragon", "sunshine", "!!@@##$$",
        "asdfghjkl", "zzzzzzzz", 
        "ThisIsALongPassword12345", "KeyboardWalk!@#$asdfghjkl",
        "MyP@ssw0rdIsVerySecure" 
    ]
    if not test_passwords_eval:
        print("No test passwords for evaluation.")
        return

    print("--- Basic Password Scoring (Markov vs. zxcvbn) ---")
    print(f"{'Password':<30} | {'Markov LogLik (n=4)':<20} | {'zxcvbn Score':<15}")
    print("-" * 70)
    
    markov_scores_all_test = []
    for pwd in test_passwords_eval:
        m_score = markov_model.get_log_likelihood(pwd)
        markov_scores_all_test.append(m_score)
        z_results = zxcvbn(pwd)
        z_score = z_results['score']
        print(f"{pwd:<30} | {m_score:<20.4f} | {z_score:<15}")
    print("-" * 70)
    markov_scores_all_test = np.array(markov_scores_all_test)

    # --- P/R/F1, ROC, PR Data ---
    # Get zxcvbn labels (True for weak, False for strong) for the test set
    y_true_labels, _ = get_zxcvbn_labels(test_passwords_eval)

    print("\n--- Precision, Recall, F1-score (Illustrative) ---")
    print("Using zxcvbn score < 3 as 'weak' (True label).")
    # For a single P/R/F1, we need a threshold for Markov scores.
    # Let's pick a threshold, e.g., median of Markov scores for passwords zxcvbn considers weak.
    # Or, more simply for demo, an arbitrary percentile or mean.
    # For this demo, let's use the median of all Markov scores as a illustrative threshold.
    # If markov_score > median_markov_score, predict weak.
    if len(markov_scores_all_test) > 0:
        # A simple illustrative threshold: median of scores of passwords zxcvbn deems weak
        weak_markov_scores = markov_scores_all_test[y_true_labels == True]
        if len(weak_markov_scores) > 0:
            illustrative_threshold = np.median(weak_markov_scores)
        else: # If no weak passwords by zxcvbn, use overall median
            illustrative_threshold = np.median(markov_scores_all_test)
            
        print(f"Illustrative Markov threshold (log-likelihood > this = predicted weak): {illustrative_threshold:.4f}")
        p, r, f1, _, _ = calculate_prf1(y_true_labels, markov_scores_all_test, illustrative_threshold)
        print(f"At this threshold: Precision={p:.4f}, Recall={r:.4f}, F1-score={f1:.4f}\n")
    else:
        print("Not enough data to calculate P/R/F1.\n")

    print("--- ROC Curve Data Points (FPR, TPR) ---")
    print("Format: (FalsePositiveRate, TruePositiveRate)")
    roc_data, pr_data = generate_roc_pr_data(y_true_labels, markov_scores_all_test)
    if roc_data:
        for i, point in enumerate(roc_data):
            print(f"Point {i+1}: ({point[0]:.4f}, {point[1]:.4f})")
        # AUC calculation (trapezoidal rule)
        auc = 0.0
        for i in range(len(roc_data) - 1):
            auc += (roc_data[i+1][0] - roc_data[i][0]) * (roc_data[i+1][1] + roc_data[i][1]) / 2.0
        print(f"Illustrative AUC (Trapezoidal Rule): {auc:.4f}")
    else:
        print("Not enough data to generate ROC points.")
    
    print("\n--- PR Curve Data Points (Recall, Precision) ---")
    print("Format: (Recall, Precision)")
    if pr_data:
        for i, point in enumerate(pr_data):
            print(f"Point {i+1}: ({point[0]:.4f}, {point[1]:.4f})")
    else:
        print("Not enough data to generate PR points.")

    # --- Approximate Guess Rank ---
    # Use a subset for cleaner output, or all test_passwords_eval
    approximate_guess_rank(markov_model, passwords_for_training, test_passwords_eval[:5]) # Demo with first 5

    # --- Policy Experiments ---
    run_policy_experiments(markov_model)
    
    print("\n" + "="*70)
    print("Demo Finished.")
    print("Reminder: These results are illustrative due to the small training dataset.")
    print("For robust metrics, a large dataset (e.g., RockYou 70k subset) is required.")
    print("="*70)


if __name__ == "__main__":
    # Path to your sample passwords file
    # Ensure this file exists and contains one password per line
    sample_password_file = "/Users/justin/Desktop/資安期末/sample_passwords.txt" 
    run_demo(sample_password_file)
