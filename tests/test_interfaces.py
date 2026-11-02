"""Contract checks for the integration infrastructure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from backend.contracts import Approximation, SearchResult
from backend.evaluation import evaluate_query, match_neighbors
from backend.external import ExternalStore, exact_neighbors
from backend.pipeline import prepare
from backend.registry import read_config


class InterfaceTests(unittest.TestCase):
    def fixture(self, root):
        np.save(root/'series.npy',np.array([[0.,0.],[1.,0.],[2.,0.],[4.,0.]],dtype=np.float32))
        np.save(root/'queries.npy',np.array([0,2],dtype=np.int64))
        config={'datasets':[{'id':'collection','data':{'path':'series.npy'},'queries':{'path':'queries.npy'}}],
                'methods':[{'id':'encoding','dimensions':2,'factory':'local:Encoder'}],
                'indexes':[{'id':'tree','factory':'local:Tree'}], 'batch_size':2}
        (root/'config.json').write_text(json.dumps(config))
        return read_config(root/'config.json')

    def test_reference_neighbors_and_matching(self):
        rows=np.array([[0.,0.],[1.,0.],[2.,0.]])
        self.assertEqual(exact_neighbors(rows[0],rows,2,0,batch_size=1),[1,2])
        self.assertEqual(match_neighbors([1,1],[1,2],rows)['score'],.5)

    def test_file_configuration_and_separate_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=self.fixture(root); store=ExternalStore(config)
            self.assertEqual(store.query_ids('collection'),[0,2])
            np.save(root/'outside.npy',np.array([[.5,0.]],dtype=np.float32))
            config['datasets'][0]['queries']={'path':'outside.npy','kind':'series'}
            self.assertEqual(store.query_ids('collection'),[-1])
            np.testing.assert_array_equal(store.query('collection',-1),[.5,0.])

    def test_pipeline_build_reuse_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.fixture(Path(tmp))
            encoder=Mock(); encoder.transform.side_effect=lambda rows:rows
            index=Mock()
            with patch('backend.pipeline.create',side_effect=lambda entry:encoder if entry['id']=='encoding' else index):
                prepare(config)
                self.assertEqual(encoder.fit.call_count,1)
                self.assertEqual(index.build.call_count,1)
                np.testing.assert_array_equal(np.load(Path(tmp)/'artifacts/collection/encoding/representations.npy'),np.load(Path(tmp)/'series.npy'))
                encoder.reset_mock(); index.reset_mock()
                prepare(config)
                self.assertEqual(encoder.fit.call_count,0)
                self.assertEqual(index.build.call_count,0)
                encoder.reset_mock(); index.reset_mock()
                prepare(config,force=True)
                self.assertEqual(index.build.call_count,1)
                encoder.reset_mock(); index.reset_mock()

    def test_invalid_encoding_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.fixture(Path(tmp)); encoder=Mock()
            encoder.transform.return_value=np.array([[float('nan')]])
            with patch('backend.pipeline.create',return_value=encoder):
                with self.assertRaises(ValueError): prepare(config)
            encoder.reset_mock()
            self.assertFalse((Path(tmp)/'artifacts/collection/encoding/manifest.json').exists())

    def test_evaluation_preserves_adapter_answers(self):
        data=np.array([[0.,0.],[1.,0.],[2.,0.]])
        models=Mock(); models.approximate.return_value=Approximation(data[0],np.array([]),np.array([]))
        indexes=Mock(); indexes.ground_truth.return_value=[1]; indexes.representation_knn.return_value=[1]
        returned=SearchResult([2],[2.],[],{'estimated_ms':1.,'candidates':1},[])
        indexes.search_curve.return_value={10.:returned}
        result=evaluate_query(data[0],0,data,{'arbitrary':{}},models,indexes,2,8,1,1e-6,(10.,),'collection')
        self.assertEqual(result['methods']['arbitrary']['coverage']['score'],1.)
        self.assertEqual(result['methods']['arbitrary']['searches'][10.]['matching']['score'],0.)
        self.assertEqual(returned.ids,[2])
        self.assertGreaterEqual(returned.metrics['estimated_ms'],0)


if __name__=='__main__': unittest.main()
