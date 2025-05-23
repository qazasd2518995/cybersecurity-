import collections
import math
import zxcvbn
import numpy as np # For AUC calculation

# --- MarkovPasswordModel Class ---
class MarkovPasswordModel:
    """
    A character-level n-gram Markov model for password strength assessment.
    """
    def __init__(self, n=4, k=1):
        """
        Initializes the Markov model.
        Args:
            n (int): The order of the n-gram model (e.g., 4 for 4-grams).
            k (int): The Laplace smoothing constant (add-k smoothing).
        """
        if n < 2:
            raise ValueError("n must be at least 2 for n-grams.")
        self.n = n
        self.k = k  # Laplace smoothing constant
        self.transitions = collections.defaultdict(lambda: collections.defaultdict(int))
        self.prefix_totals = collections.defaultdict(int)
        self.vocabulary = set()
        self.start_token = '^'
        self.end_token = '$'
        self.trained = False

    def _pad_password(self, password):
        """Pads password with start and end tokens."""
        # Pad with n-1 start tokens
        return (self.start_token * (self.n - 1)) + password + self.end_token

    def train(self, passwords_list):
        """
        Trains the Markov model on a list of passwords.
        Args:
            passwords_list (list): A list of password strings.
        """
        if not passwords_list:
            print("Warning: Training with an empty password list.")
            return

        for password in passwords_list:
            if not isinstance(password, str):
                print(f"Warning: Skipping non-string item in password list: {password}")
                continue
            
            padded_password = self._pad_password(password)
            self.vocabulary.update(set(padded_password))

            for i in range(len(padded_password) - (self.n - 1)):
                prefix = padded_password[i : i + self.n - 1]
                next_char = padded_password[i + self.n - 1]
                
                self.transitions[prefix][next_char] += 1
                self.prefix_totals[prefix] += 1
        
        self.trained = True
        # Add special tokens to vocabulary if not present
        self.vocabulary.add(self.start_token)
        self.vocabulary.add(self.end_token)


    def get_log_likelihood(self, password):
        """
        Computes the log-likelihood of a given password under the model.
        A higher log-likelihood (closer to 0) means a weaker password.
        Args:
            password (str): The password string to score.
        Returns:
            float: The log-likelihood of the password. Returns -float('inf')
                   if the model is untrained or vocabulary is empty.
        """
        if not self.trained or not self.vocabulary:
            # print("Warning: Model is not trained or vocabulary is empty. Returning -inf.")
            return -float('inf')

        if not isinstance(password, str):
             # print(f"Warning: Input password is not a string: {password}. Returning -inf.")
             return -float('inf')

        padded_password = self._pad_password(password)
        log_likelihood = 0.0
        vocab_size = len(self.vocabulary)

        for i in range(len(padded_password) - (self.n - 1)):
            prefix = padded_password[i : i + self.n - 1]
            next_char = padded_password[i + self.n - 1]

            count_prefix_next_char = self.transitions[prefix].get(next_char, 0)
            count_prefix_total = self.prefix_totals.get(prefix, 0)

            # Laplace smoothing (add-k)
            prob = (count_prefix_next_char + self.k) / (count_prefix_total + self.k * vocab_size)
            
            if prob == 0: # Should not happen with k > 0 if vocab_size > 0
                log_likelihood = -float('inf')
                break 
            log_likelihood += math.log(prob)
            
        return log_likelihood

# --- Helper Functions ---
def get_zxcvbn_details(password):
    """
    Gets zxcvbn score and a weak/strong label for a password.
    Args:
        password (str): The password to evaluate.
    Returns:
        tuple: (zxcvbn_score (0-4), is_weak (bool: True if score < 3))
    """
    if not isinstance(password, str): # Handle non-string inputs gracefully
        return -1, True # Or raise an error, depending on desired behavior
    results = zxcvbn.zxcvbn(password)
    score = results['score']
    is_weak = score < 3  # 0, 1, 2 are weak; 3, 4 are strong
    return score, is_weak

def calculate_classification_metrics(true_labels, pred_scores, threshold):
    """
    Calculates Precision, Recall, and F1-score.
    Args:
        true_labels (list): List of true boolean labels (True for weak).
        pred_scores (list): List of Markov log-likelihood scores.
        threshold (float): Markov score threshold to classify as weak.
                           (score > threshold is predicted weak).
    Returns:
        tuple: (precision, recall, f1_score)
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    for true_label, score in zip(true_labels, pred_scores):
        predicted_weak = score > threshold
        if predicted_weak and true_label:
            tp += 1
        elif predicted_weak and not true_label:
            fp += 1
        elif not predicted_weak and not true_label:
            tn += 1
        elif not predicted_weak and true_label:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score

def get_roc_pr_data(true_labels, pred_scores):
    """
    Generates data points for ROC and PR curves and calculates AUC for ROC.
    Args:
        true_labels (list): List of true boolean labels (True for weak/positive).
        pred_scores (list): List of prediction scores (higher score = more likely positive/weak).
    Returns:
        tuple: (roc_points, auc, pr_points)
               roc_points: list of (FPR, TPR) tuples
               auc: Area Under ROC Curve
               pr_points: list of (Recall, Precision) tuples
    """
    # Sort by prediction score in descending order
    # If scores are log-likelihoods, higher means weaker, so this is correct for "weak" as positive
    combined = sorted(zip(pred_scores, true_labels), key=lambda x: x[0], reverse=True)
    
    roc_points = []
    pr_points = []
    
    total_positives = sum(true_labels)
    total_negatives = len(true_labels) - total_positives

    if total_positives == 0 or total_negatives == 0: # Cannot compute ROC/PR if only one class present
        # print("Warning: Only one class present in true_labels. ROC/PR/AUC cannot be computed meaningfully.")
        return [(0,0), (1,1)], 0.5, [(0,1), (1,0)] if total_positives > 0 else [(0,0),(1,0)]


    tp, fp = 0, 0
    
    # Add a point for the beginning of the curve
    roc_points.append((0.0, 0.0))
    # For PR curve, initial point depends on the first threshold.
    # Typically, PR curve starts from (Recall=0, Precision=1) if the highest score is a TP.
    # Or (Recall=0, Precision= undefined/0) if highest score is FP.
    # Let's calculate it dynamically.

    last_score = -float('inf')

    for i in range(len(combined)):
        score, true_label = combined[i]

        if score != last_score: # Only update points when threshold changes
            if total_negatives > 0:
                fpr = fp / total_negatives
            else: # Should be caught by earlier check
                fpr = 0 
            
            if total_positives > 0:
                tpr = tp / total_positives # Recall
            else: # Should be caught by earlier check
                tpr = 0
            
            roc_points.append((fpr, tpr))

            if (tp + fp) > 0:
                precision = tp / (tp + fp)
                pr_points.append((tpr, precision)) # (Recall, Precision)
            elif tp == 0 and total_positives > 0: # No predictions made yet, or all false
                 pr_points.append((tpr, 0.0))


            last_score = score

        if true_label: # Positive
            tp += 1
        else: # Negative
            fp += 1

    # Add a point for the end of the curve
    if total_negatives > 0:
        fpr = fp / total_negatives
    else:
        fpr = 1.0 # Effectively
    
    if total_positives > 0:
        tpr = tp / total_positives
    else:
        tpr = 1.0 # Effectively

    roc_points.append((fpr, tpr)) # Should be (1,1) if all points processed

    if (tp + fp) > 0 :
        precision = tp / (tp + fp)
        pr_points.append((tpr, precision))
    elif tp == 0 and total_positives > 0:
        pr_points.append((tpr,0.0))
    else: # Handle case where no positives exist or no TPs were found.
        pr_points.append((1.0, 0.0 if total_positives > 0 else 1.0))


    # Remove duplicate points for cleaner curves
    roc_points = sorted(list(set(roc_points)))
    pr_points = sorted(list(set(pr_points)))


    # Calculate AUC using trapezoidal rule
    auc = 0.0
    if len(roc_points) > 1:
        # Ensure roc_points are sorted by FPR
        roc_points.sort(key=lambda x: x[0])
        for i in range(len(roc_points) - 1):
            auc += (roc_points[i+1][0] - roc_points[i][0]) * (roc_points[i+1][1] + roc_points[i][1]) / 2.0
    
    return roc_points, auc, pr_points

def approximate_guess_rank(model, all_passwords, test_passwords_to_rank):
    """
    Computes approximate guess rank for specified test passwords.
    Args:
        model: The trained MarkovPasswordModel.
        all_passwords (list): A list of all passwords (training + test) to rank.
        test_passwords_to_rank (list): A sublist of passwords whose ranks are desired.
    Returns:
        list: A list of tuples (password, rank, log_likelihood).
    """
    scored_passwords = []
    for p in all_passwords:
        if isinstance(p, str): # ensure it's a string
            scored_passwords.append((p, model.get_log_likelihood(p)))

    # Sort by log-likelihood (descending: higher score = weaker = earlier guess)
    scored_passwords.sort(key=lambda x: x[1], reverse=True)

    ranks = []
    for p_test in test_passwords_to_rank:
        if not isinstance(p_test, str): continue
        found = False
        for i, (p_ranked, score) in enumerate(scored_passwords):
            if p_test == p_ranked:
                ranks.append((p_test, i + 1, score)) # Rank is 1-indexed
                found = True
                break
        if not found: # Should not happen if test_passwords_to_rank is subset of all_passwords
            ranks.append((p_test, -1, model.get_log_likelihood(p_test))) # Or handle as error
    return ranks

def run_policy_experiments(model, zxcvbn_func):
    """
    Runs 'what-if' experiments for password policy transformations.
    Args:
        model: The trained MarkovPasswordModel.
        zxcvbn_func: Function to get zxcvbn score (e.g., lambda p: get_zxcvbn_details(p)[0]).
    """
    print("--- Password Policy Experiments ---")
    print(f"{'Policy':<30} | {'Variation':<15} | {'Password':<25} | {'Markov LogLik (n=4)':<20} | {'zxcvbn Score':<15}")
    print("-" * 120)

    experiments = {
        "Length Increase": [
            ("base", "apple"),
            ("base+1", "apple1"),
            ("base+2", "apple12"),
            ("base+3", "apple123"),
            ("base+long", "appleorangebanana")
        ],
        "Symbol Inclusion / Substitution": [
            ("plain", "password"),
            ("capitalized", "Password"),
            ("leet_simple", "P@ssword"),
            ("leet_complex", "P@$$wOrd"), # From report
            ("with_symbols", "password!@#")
        ]
    }

    for policy_name, variations in experiments.items():
        for variation_name, pwd in variations:
            if not isinstance(pwd, str): continue
            markov_score = model.get_log_likelihood(pwd)
            z_score, _ = zxcvbn_func(pwd)
            print(f"{policy_name:<30} | {variation_name:<15} | {pwd:<25} | {markov_score:<20.4f} | {z_score:<15}")
        print("-" * 120)

# --- Main Demo Function ---
def run_demo(password_file_path):
    """
    Orchestrates the entire demo.
    """
    print("=" * 70)
    print("Machine Learning for Password Strength: Markov Model Demo (n=4)")
    print("=" * 70)
    print("IMPORTANT NOTE:")
    print("This demo uses a VERY SMALL training dataset for illustrative purposes.")

    # 1. Model Training
    try:
        with open(password_file_path, 'r', encoding='utf-8') as f:
            # Filter out empty lines and strip whitespace
            training_passwords = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Password file '{password_file_path}' not found.")
        print("Please create 'sample_passwords.txt' or specify the correct path.")
        # Fallback to a small embedded list for demo purposes if file not found
        training_passwords = [
            "123456", "password", "qwerty", "admin", "P@$$wOrd", "apple", "secret123",
            "dragon", "sunshine", "!!@@##$$", "asdfghjkl", "zzzzzzzz", 
            "ThisIsALongPassword12345", "KeyboardWalk!@#$asdfghjkl", "MyP@sswordIsVerySecure",
            "P@$$word", # as in report
            "complexP@ss!", "anotherOne", "testtest", "user123", "Spring2024",
            "football", "shadow", "monkey", "cheese", "remember", "master"
        ]
        print(f"Using a fallback list of {len(training_passwords)} passwords for training.")
        
    print(f"({len(training_passwords)} passwords from {password_file_path if 'f' in locals() else 'fallback list'}).")

    markov_model = MarkovPasswordModel(n=4, k=1)
    markov_model.train(training_passwords)
    print(f"Model training complete. Vocabulary size: {len(markov_model.vocabulary)}")

    # 2. Basic Scoring Comparison
    print("\n--- Basic Password Scoring (Markov vs. zxcvbn) ---")
    print(f"{'Password':<28} | {'Markov LogLik (n=4)':<20} | {'zxcvbn Score':<15} | {'zxcvbn Weak':<10}")
    print("-" * 82)
    
    test_passwords_eval = [
        "123456", "password", "qwerty", "admin",
        "P@$$wOrd", # Report uses P@$$wOrd (capital O)
        "StrongP@ssword!", "secret123", "dragon", "sunshine",
        "!!@@##$$", "asdfghjkl", "zzzzzzzz",
        "ThisIsALongPassword12345",
        "KeyboardWalk!@#$asdfghjkl", # From report
        "MyP@sswordIsVerySecure" # Report example typo: MyP@ssw0rdIsVerySecure, using report's table version
    ]
    # Add some more diverse cases
    test_passwords_eval.extend([
        "apple", "orange123", "Complex!", "verystrongandlongpassword", "111111", "abcdef"
    ])
    test_passwords_eval = sorted(list(set(test_passwords_eval + training_passwords[:5]))) # include some training for varied scores


    zxcvbn_true_labels_weak = [] # True if zxcvbn score < 3
    markov_pred_scores = []

    for pwd in test_passwords_eval:
        if not isinstance(pwd, str): continue
        m_score = markov_model.get_log_likelihood(pwd)
        z_score, z_weak = get_zxcvbn_details(pwd)
        print(f"{pwd:<28} | {m_score:<20.4f} | {z_score:<15} | {str(z_weak):<10}")
        zxcvbn_true_labels_weak.append(z_weak)
        markov_pred_scores.append(m_score)
    print("-" * 82)

    # 3. Precision/Recall/F1 Calculation
    print("\n--- Precision, Recall, F1-score (Illustrative) ---")
    print("Using zxcvbn score < 3 as 'weak' (True label).")

    # Determine threshold: median Markov log-likelihood among passwords zxcvbn labeled as weak
    weak_passwords_markov_scores = []
    for pwd in training_passwords: # Use training passwords to set threshold, as per typical ML practice
        if not isinstance(pwd, str): continue
        z_score, z_weak = get_zxcvbn_details(pwd)
        if z_weak:
            weak_passwords_markov_scores.append(markov_model.get_log_likelihood(pwd))
    
    if weak_passwords_markov_scores:
        threshold = np.median(weak_passwords_markov_scores)
    else: # Fallback if no weak passwords in training by zxcvbn's standard
        if markov_pred_scores:
            threshold = np.median(markov_pred_scores) 
        else:
            threshold = -30.0 # Arbitrary fallback
        print("Warning: No passwords labeled as weak by zxcvbn in training set to determine median threshold, using overall median or fallback.")


    print(f"Illustrative Markov threshold (log-likelihood > {threshold:.4f} = predicted weak)")
    
    if zxcvbn_true_labels_weak and markov_pred_scores:
        precision, recall, f1 = calculate_classification_metrics(zxcvbn_true_labels_weak, markov_pred_scores, threshold)
        print(f"At this threshold: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")
    else:
        print("Not enough data to calculate P/R/F1.")

    # 4. ROC and PR Curve Data
    print("\n--- ROC Curve Data Points (FPR, TPR) ---")
    if zxcvbn_true_labels_weak and markov_pred_scores:
        roc_points, auc, pr_points = get_roc_pr_data(zxcvbn_true_labels_weak, markov_pred_scores)
        print(roc_points)
        print(f"Illustrative AUC (Trapezoidal Rule): {auc:.4f}")
        print("\n--- PR Curve Data Points (Recall, Precision) ---")
        print(pr_points)
    else:
        print("Not enough data for ROC/PR curves.")


    # 5. Approximate Guess Number Estimation
    print("\n--- Approximate Guess Rank ---")
    print("Note: Based on sorting all training + test passwords by Markov log-likelihood.")
    print("Lower rank (closer to 1) means model finds it weaker/more guessable.")
    
    all_pswds_for_ranking = list(set(training_passwords + test_passwords_eval))
    # Select a few passwords to show rank for, e.g., the first few from test_passwords_eval
    passwords_to_get_rank_for = [p for p in test_passwords_eval[:5] if isinstance(p,str)] 
    
    if passwords_to_get_rank_for:
        ranked_list = approximate_guess_rank(markov_model, all_pswds_for_ranking, passwords_to_get_rank_for)
        print(f"{'Test Password':<28} | {'Approx. Guess Rank':<20} | {'Markov LogLikelihood':<20}")
        print("-" * 75)
        for pwd, rank, score in ranked_list:
            print(f"{pwd:<28} | {rank:<20} | {score:<20.4f}")
        print(f"Total passwords considered for ranking: {len(all_pswds_for_ranking)}")
    else:
        print("No passwords selected for rank demonstration.")


    # 6. Password Policy Experiments
    print() # newline
    run_policy_experiments(markov_model, get_zxcvbn_details)

    print("=" * 70)
    print("Demo Finished.")
    print("Reminder: These results are illustrative due to the small training dataset.")
    print("For robust metrics, a large dataset (e.g., RockYou 70k subset) is required.")
    print("=" * 70)


if __name__ == "__main__":
    # IMPORTANT: User should create 'sample_passwords.txt' in the same directory
    # or update this path to the correct location.
    sample_password_file = "sample_passwords.txt" 
    run_demo(sample_password_file)
