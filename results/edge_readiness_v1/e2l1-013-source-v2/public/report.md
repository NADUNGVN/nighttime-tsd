# L1A-012 source bundle

Status: execution_complete_numerical_fail; numerical verdict: fail.

{
  "commit": "7dd178af4c001d074d1efbd4bbbafcb767dc6263",
  "counters": {
    "fixture_images_attempted": 3,
    "fixture_images_verified": 3,
    "inputs_attempted": 3,
    "inputs_completed": 3,
    "native_forwards_attempted": 3,
    "native_forwards_completed": 3,
    "export_invocations_attempted": 1,
    "export_invocations_completed": 1,
    "exporter_internal_forwards": 4,
    "ort_forwards_attempted": 3,
    "ort_forwards_completed": 3,
    "comparisons_attempted": 3,
    "comparisons_completed": 3
  },
  "comparisons": {
    "00006": {
      "status": "pass",
      "equation": "abs(observed-reference) <= 1e-5 + 1e-4*abs(reference)",
      "elements": 58800,
      "failing_elements": 0,
      "box_failing_elements": 0,
      "score_failing_elements": 0,
      "max_abs_error": 0.0016632080078125,
      "box_max_abs_error": 0.0016632080078125,
      "score_max_abs_error": 2.980232238769531e-07,
      "max_relative_error": 5.84876910707276,
      "bounded_failures": []
    },
    "00009": {
      "status": "pass",
      "equation": "abs(observed-reference) <= 1e-5 + 1e-4*abs(reference)",
      "elements": 58800,
      "failing_elements": 0,
      "box_failing_elements": 0,
      "score_failing_elements": 0,
      "max_abs_error": 0.0003509521484375,
      "box_max_abs_error": 0.0003509521484375,
      "score_max_abs_error": 1.1324882507324219e-06,
      "max_relative_error": 6.758800016694825,
      "bounded_failures": []
    },
    "00028": {
      "status": "fail",
      "equation": "abs(observed-reference) <= 1e-5 + 1e-4*abs(reference)",
      "elements": 58800,
      "failing_elements": 1,
      "box_failing_elements": 1,
      "score_failing_elements": 0,
      "max_abs_error": 0.000335693359375,
      "box_max_abs_error": 0.000335693359375,
      "score_max_abs_error": 2.384185791015625e-07,
      "max_relative_error": 6.365596604765779,
      "bounded_failures": [
        {
          "flat_index": 16402,
          "reference": 0.002925872802734375,
          "observed": 0.002941131591796875,
          "abs_error": 1.52587890625e-05,
          "limit": 1.0292587280273439e-05
        }
      ]
    }
  }
}
