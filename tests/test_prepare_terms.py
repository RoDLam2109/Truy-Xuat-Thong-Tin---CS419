import json
from pathlib import Path
import tempfile
import unittest
from scripts.prepare_terms import normalize, char_terms, word_terms, terms_for, prepare


class TermTests(unittest.TestCase):
    def test_normalization_preserves_noise_numbers_and_negation(self):
        self.assertEqual(normalize('  KHÔNG\n đc  12/2024! '), 'không đc 12/2024!')
        self.assertEqual(normalize(normalize('Á  B')), normalize('Á  B'))
        words = word_terms(normalize('Không được và phải có 123.'), {'và'})
        self.assertIn('không', words)
        self.assertIn('123', words)
        self.assertNotIn('và', words)

    def test_character_boundaries_and_filter_keep_counts(self):
        grams = list(char_terms('phạt xe'))
        self.assertIn('phạ', grams)
        self.assertIn('ạt ', grams)
        self.assertNotIn('t x', grams)
        row = {'text': 'xe xe', 'word_terms': ['xe', 'xe', 'lạ']}
        self.assertEqual(terms_for(row, 'word', {'xe': 2}), ['xe', 'xe'])
        self.assertEqual(terms_for(row, 'char_ngram', {' xe ': 2}), [' xe ', ' xe '])

    def test_vocabulary_fits_only_corpus_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'data', Path(tmp) / 'out'
            (source / 'noise').mkdir(parents=True)
            docs = [{'id': 'd1', 'title': '', 'text': 'xe xe'},
                    {'id': 'd2', 'title': '', 'text': 'xe tàu'}]
            (source / 'corpus.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in docs))
            query = json.dumps({'id': 'q1', 'text': 'unseenquery'}) + '\n'
            for name in ['queries_train.jsonl', 'queries_clean.jsonl']:
                (source / name).write_text(query)
            for i in range(7):
                (source / 'noise' / f'queries_{i}.jsonl').write_text(query)
            for split in ['train', 'test']:
                (source / f'qrels_{split}.tsv').write_text('query-id\tcorpus-id\tscore\nq1\td1\t1\n')
            before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
            prepare(source, output, {'stopwords': [], 'min_df': 1, 'max_df': 1.0, 'ngram_range': [3, 5]}, workers=2)
            vocab = json.loads((output / 'vocabulary_word.json').read_text())
            self.assertEqual(vocab['xe'], 2)  # DF counts documents, not repeated occurrences.
            self.assertNotIn('unseenquery', vocab)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with self.assertRaises(ValueError):
                prepare(source, source, {})


if __name__ == '__main__':
    unittest.main()
