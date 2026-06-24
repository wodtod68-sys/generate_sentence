==============================================================================
 Bigram (2-gram) Statistical Language Model — Training & Sentence Generation
==============================================================================

A minimal, from-scratch statistical language model that learns word-level
bigram statistics from a plain-text corpus and then uses them to (a) generate
new sentences and (b) score the likelihood of a sentence. No external NLP or
machine-learning libraries are used — the model is built purely from frequency
counts.

The project has two stages:
  learn_lm.py          -> trains the model and serializes it
  generate_sentence.py -> loads the model, generates and ranks sentences


------------------------------------------------------------------------------
1. MODELING APPROACH
------------------------------------------------------------------------------

The model is a first-order Markov (bigram) language model. It assumes the
probability of a word depends only on the single preceding word:

    P(w_i | w_1 ... w_{i-1})  ~  P(w_i | w_{i-1})

Probabilities are estimated by Maximum Likelihood Estimation (MLE) directly from
raw counts:

    P(w2 | w1) = count(w1, w2) / count(w1)

Two special boundary tokens frame every sentence:
  <s>   sentence-start marker
  </s>  sentence-end marker

These let the model learn which words tend to begin a sentence (bigrams from
<s>) and which tend to end it (bigrams into </s>), and they give generation a
well-defined start and stop condition.

Tokenization is whitespace-based: each line of the corpus is treated as one
sentence and split on spaces. This keeps the pipeline language-agnostic (it
works on any space-separated, pre-tokenized text).


------------------------------------------------------------------------------
2. TRAINING  (learn_lm.py)
------------------------------------------------------------------------------

Input : a text corpus, one sentence per line.
Output: a pickled model dictionary.

Procedure:
  1. Read the corpus line by line; skip empty lines.
  2. Wrap each sentence as  [<s>] + tokens + [</s>].
  3. Accumulate two count tables in a single pass:
       - unigram_counts : frequency of each token  (dict: word -> count)
       - bigram_counts  : frequency of each adjacent pair, stored as a
                          dictionary-of-dictionaries
                          (dict: w1 -> { w2 -> count(w1, w2) })
  4. Convert the nested defaultdicts to plain dicts for safe serialization.
  5. Save the model as a pickle file containing:
       { 'unigram_counts': ..., 'bigram_counts': ... }

Storing counts (rather than pre-computed probabilities) keeps the model
flexible: probabilities are derived on demand at generation/scoring time.


------------------------------------------------------------------------------
3. SENTENCE GENERATION & SCORING  (generate_sentence.py)
------------------------------------------------------------------------------

The generator exposes three core routines.

(a) get_next_word(model, current_word) — frequency-weighted sampling
    Given the current word, the next word is drawn by WEIGHTED random sampling
    over its observed successors, using the bigram counts as weights
    (random.choices). More frequent transitions are more likely to be chosen,
    so generation reflects the corpus statistics rather than picking the single
    most likely word every time.
    Fallback: if the current word was never seen as a left context, a word is
    drawn uniformly at random from the unigram vocabulary; if the vocabulary is
    empty, </s> is emitted to terminate.

(b) generate_sentence(model, start_with) — Markov-chain rollout
    Generation begins from <s> alone, or from <s> + user-supplied seed words.
    Words are appended one at a time via get_next_word until either the </s>
    token is produced or a safety cap of 50 tokens is reached (to prevent
    runaway loops). The <s>/</s> markers are stripped from the final output.

(c) get_probability(model, sentence) — log-likelihood scoring
    A sentence is re-tokenized as [<s>] + words + [</s>], and its total log
    probability is the sum of the log bigram probabilities:

        log P(sentence) = SUM over i of  log P(w_{i+1} | w_i)

    Working in log space turns a product of small probabilities into a stable
    sum and avoids floating-point underflow. Any bigram not seen during
    training (an unknown word or an unseen transition) incurs a fixed penalty
    of -100 per occurrence, sharply down-ranking sentences that the model
    considers implausible. This is a simple penalty in place of formal
    smoothing.


------------------------------------------------------------------------------
4. INTERACTIVE LOOP & RANKING
------------------------------------------------------------------------------

generate_sentence.py runs an interactive prompt:
  - The user presses Enter (free generation) or types one or more seed words to
    fix the sentence start; 'q' quits.
  - For each request the program generates 10 candidate sentences, scores each
    with get_probability, and SORTS them by log probability in descending order.
  - Results are printed best-first, each with its log-probability, so the most
    statistically coherent candidates surface at the top.

This sample-then-rank scheme combines the diversity of stochastic sampling with
a likelihood-based quality filter.


------------------------------------------------------------------------------
5. DESIGN NOTES / LIMITATIONS
------------------------------------------------------------------------------

  - First-order context only: a bigram model has very short memory, so longer-
    range grammatical structure is not captured.
  - No smoothing: unseen bigrams rely on the -100 penalty rather than a
    principled technique (e.g. add-k or Kneser-Ney), so the scores are
    comparative rather than true normalized probabilities.
  - Vocabulary and tokenization are defined entirely by the corpus and its
    whitespace segmentation; quality of output scales with corpus size and
    cleanliness.


------------------------------------------------------------------------------
6. REPOSITORY CONTENTS
------------------------------------------------------------------------------

  learn_lm.py            Trains the bigram model from a corpus and pickles it.
  generate_sentence.py   Loads the model; interactively generates and ranks
                         sentences, and scores sentence log-probabilities.

==============================================================================
