import copy
import unittest
from scripts.export_submission import actual_value, score_from_values, source_records, validate_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.source = [{'conversation_id':'unseen-control', 'transcript':[
            {'speaker':'system','text':'Start'}, {'speaker':'client','text':'Help'},
            {'speaker':'operator','text':'Hello, I am Eva.'}]}]
        values = {'contact':True, 'discovery':True, 'identification':None,
                  'consultation':15, 'empathy':True, 'dialogue':7, 'closure':True, 'critical':True}
        criteria = [{'criterion_id':key,'value':value,'rationale':'Synthetic only', 'evidence_turns':[3]}
                    for key,value in values.items()]
        self.payload = {'schema_version':'1.0','team_name':'Fixture',
            'model':{'mode':'local','name':'synthetic fixture'}, 'results':[
                {'conversation_id':'unseen-control','status':'ok','criteria':criteria,
                 'score':score_from_values(criteria)}]}

    def test_original_turns_and_na_weight_and_critical_do_not_zero(self):
        self.assertEqual(self.payload['results'][0]['score'], {'final_score':88.13,'critical_flag':True})
        self.assertEqual(validate_submission(self.payload,self.source,1)['ok'],1)

    def test_bool_not_accepted_as_numeric_scale(self):
        self.payload['results'][0]['criteria'][3]['value'] = True
        with self.assertRaises(ValueError): validate_submission(self.payload,self.source,1)

    def test_wrong_flag_or_score_rejected(self):
        for key,value in [('critical_flag',False),('final_score',0)]:
            changed=copy.deepcopy(self.payload);changed['results'][0]['score'][key]=value
            with self.assertRaises(ValueError):validate_submission(changed,self.source,1)

    def test_missing_duplicate_and_unknown_ids_rejected(self):
        for changed in [[], self.payload['results']*2,
                        [{**self.payload['results'][0], 'conversation_id':'swap-unwanted'}]]:
            payload=copy.deepcopy(self.payload);payload['results']=changed
            with self.assertRaises(ValueError):validate_submission(payload,self.source,1)

    def test_out_of_bounds_zero_based_turn_rejected(self):
        for turns in [[0],[4],[True],[3,3]]:
            self.payload['results'][0]['criteria'][0]['evidence_turns']=turns
            with self.assertRaises(ValueError):validate_submission(self.payload,self.source,1)

    def test_error_is_retained_and_counts_toward_exact_coverage(self):
        self.payload['results']=[{'conversation_id':'unseen-control','status':'error','criteria':[],
                                 'score':None,'error':'Provider timeout'}]
        self.assertEqual(validate_submission(self.payload,self.source,1),{'records':1,'ok':0,'errors':1})

    def test_wrong_null_or_string_values_rejected(self):
        for value in [None,'true',1]:
            self.payload['results'][0]['criteria'][0]['value']=value
            with self.assertRaises(ValueError):validate_submission(self.payload,self.source,1)

    def test_duplicate_source_and_declared_count_rejected(self):
        with self.assertRaises(ValueError):source_records(self.source*2)
        with self.assertRaises(ValueError):source_records({'data':{'records':self.source,'total_records':2}})

    def test_internal_answers_converted_without_coercing_arbitrary_numbers(self):
        self.assertIs(actual_value('yes','contact'),True)
        self.assertIs(actual_value('no','critical'),False)
        self.assertIsNone(actual_value('na','discovery'))
        with self.assertRaises(ValueError): actual_value('8','dialogue')
        with self.assertRaises(ValueError): actual_value('na','critical')


if __name__=='__main__':unittest.main()
