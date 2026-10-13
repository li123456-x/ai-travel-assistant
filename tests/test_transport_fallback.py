"""No live Amap or model requests: exercise the actual transport functions."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import test_transport_nodes as transport_fixture


class TransportFallbackTests(unittest.TestCase):
    def setUp(self):
        fixture = transport_fixture.TransportNodeTests(methodName="test_default_modes_and_walking_are_supported")
        fixture.setUp()
        self.fixture = fixture
        self.state = {
            **fixture.state, "days": 1, "return_date": fixture.state["departure_date"],
            "hotel_name": "", "attraction_names": ["西安城墙", "大雁塔"],
        }
        fixture.geocoding.return_value = ({
            "西安城墙": {"location": "108.95,34.25", "citycode": "029"},
            "大雁塔": {"location": "108.96,34.24", "citycode": "029"},
            "陕西历史博物馆": {"location": "108.97,34.23", "citycode": "029"},
            "回民街": {"location": "108.98,34.22", "citycode": "029"},
        }, {})

    def run_node(self, **updates):
        return self.fixture.namespace["transport_node"]({**self.state, **updates})

    def empty_transit_then_walking(self, *, distance="1000", duration="800"):
        def query(args):
            if args["mode"] == "transit":
                return '{"status":"1","route":{"transits":[]}}'
            path = {"distance": distance}
            if duration is not None:
                path["cost"] = {"duration": duration}
            return json.dumps({"status": "1", "route": {"paths": [path]}})
        self.fixture.routes.side_effect = query

    def test_short_empty_transit_segment_uses_real_walking_metrics(self):
        self.empty_transit_then_walking()
        result = self.run_node()
        self.assertEqual([call.args[0]["mode"] for call in self.fixture.routes.call_args_list], ["transit", "walking"])
        segment = result["transport_segments"][0]
        self.assertEqual(segment["requested_mode"], "transit")
        self.assertEqual(segment["actual_mode"], "walking")
        self.assertEqual(segment["distance_m"], 1000)
        self.assertEqual(segment["duration_s"], 800)
        self.assertEqual(self.fixture.geocoding.call_args.kwargs["places"], ["西安城墙", "大雁塔"])

    def test_route_requests_ask_for_cost_duration_for_all_supported_modes(self):
        response = Mock()
        response.json.return_value = {"status": "1", "route": {"paths": [{"distance": "800"}]}}
        get = Mock(return_value=response)
        namespace = transport_fixture.load_functions("tools/transport.py", {
            "AMAP_KEY": "offline-test-key", "json": json,
            "requests": SimpleNamespace(get=get, RequestException=RuntimeError),
        })
        namespace["_wait_for_route_request"] = Mock()
        for mode in ("transit", "walking", "driving"):
            with self.subTest(mode=mode):
                namespace["_request_route"]("108.95,34.25", "108.96,34.24", "029", "029", mode)
                self.assertEqual(get.call_args.kwargs["params"]["show_fields"], "cost")
                self.assertEqual(get.call_args.kwargs["timeout"], 10)

    def test_one_day_single_attraction_needs_only_outbound_and_return_queries(self):
        result = self.run_node(attraction_names=["西安城墙"])
        self.assertEqual(self.fixture.intercity.call_count, 2)
        self.fixture.geocoding.assert_not_called()
        self.fixture.routes.assert_not_called()
        self.assertEqual(result["transport_segments"], [])

    def test_disabled_walking_fallback_stops_before_walking_api(self):
        self.empty_transit_then_walking()
        with self.assertRaisesRegex(ValueError, "未允许"):
            self.run_node(allow_walking_fallback=False)
        self.fixture.routes.assert_called_once()
        self.fixture.model.assert_not_called()

    def test_api_failures_malformed_json_and_missing_route_are_not_empty_results(self):
        for raw in ("交通查询请求失败：timeout", "not json", '{"status":"0"}',
                    '{"status":"1","route":{}}', '{"status":"1","route":{"transits":[{}]}}'):
            with self.subTest(raw=raw):
                self.fixture.routes.reset_mock()
                self.fixture.routes.side_effect = None
                self.fixture.routes.return_value = raw
                with self.assertRaises(ValueError):
                    self.run_node()
                self.fixture.routes.assert_called_once()
                self.fixture.model.assert_not_called()

    def test_walking_limits_reject_excess_distance_time_or_missing_metrics(self):
        for distance, duration in (("2500", "900"), ("1000", "2000"), ("1000", None),
                                   ("NaN", "900"), ("-1", "900")):
            with self.subTest(distance=distance, duration=duration), self.assertRaises(ValueError):
                self.empty_transit_then_walking(distance=distance, duration=duration)
                self.run_node()
        self.fixture.model.assert_not_called()

    def test_exact_single_segment_bounds_are_accepted(self):
        self.empty_transit_then_walking(distance="2000", duration="1800")
        self.assertEqual(self.run_node()["transport_segments"][0]["actual_mode"], "walking")

    def test_total_walking_limit_blocks_multiple_individually_valid_segments(self):
        self.empty_transit_then_walking(distance="1600", duration="900")
        with self.assertRaisesRegex(ValueError, "累计"):
            self.run_node(attraction_names=["西安城墙", "大雁塔", "陕西历史博物馆", "回民街"])
        self.fixture.model.assert_not_called()

    def test_successful_transit_and_walking_fallback_can_coexist(self):
        def query(args):
            if args["mode"] == "walking":
                return '{"status":"1","route":{"paths":[{"distance":"800","cost":{"duration":"600"}}]}}'
            if args["origin_location"] == "108.96,34.24":
                return '{"status":"1","route":{"transits":[]}}'
            return '{"status":"1","route":{"transits":[{"distance":"3000","cost":{"duration":"900"}}]}}'
        self.fixture.routes.side_effect = query
        result = self.run_node(attraction_names=["西安城墙", "大雁塔", "陕西历史博物馆"])
        self.assertEqual([item["actual_mode"] for item in result["transport_segments"]], ["transit", "walking"])


if __name__ == "__main__":
    unittest.main()
