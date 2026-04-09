import requests

class MapEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.geo_url = "https://restapi.amap.com/v3/geocode/geo"
        self.route_url = "https://restapi.amap.com/v3/direction/driving"
        self.tips_url = "https://restapi.amap.com/v3/assistant/inputtips"

    # --- 功能 3：输入提示 (Input Tips) ---
    def get_input_tips(self, keywords, city):
        """输入关键词，返回高德建议的精准 POI 名称列表"""
        if not keywords: return []
        params = {
            "key": self.api_key,
            "keywords": keywords,
            "city": city,
            "datatype": "poi"
        }
        try:
            res = requests.get(self.tips_url, params=params).json()
            if res["status"] == "1":
                # 过滤掉没有具体名称的提示
                return [tip["name"] for tip in res["tips"] if isinstance(tip["name"], str)]
        except:
            pass
        return []

    def get_coords(self, address, city):
        """地址 -> 经纬度"""
        params = {"key": self.api_key, "address": address, "city": city}
        try:
            res = requests.get(self.geo_url, params=params, timeout=5).json()
            if res["status"] == "1" and res["geocodes"]:
                return res["geocodes"][0]["location"]
        except:
            pass
        return None

    def get_route_info(self, origin, destination):
        """精准路网计算：距离 + 时间"""
        params = {
            "key": self.api_key,
            "origin": origin,
            "destination": destination,
            "extensions": "base"  # 简化返回，只拿基础路线信息
        }
        try:
            res = requests.get(self.route_url, params=params, timeout=5).json()
            if res["status"] == "1" and "route" in res:
                path = res["route"]["paths"][0]
                return {
                    "km": round(int(path["distance"]) / 1000, 2),
                    "min": round(int(path["duration"]) / 60, 1)
                }
        except:
            pass
        return None

    def optimize_route(self, city, start_name, poi_list):
        """物理路径排序逻辑"""
        all_points = [start_name] + poi_list
        coords_map = {p: self.get_coords(p, city) for p in all_points if self.get_coords(p, city)}

        ordered_route = [start_name]
        current_name = start_name
        remaining_names = [p for p in poi_list if p in coords_map]
        route_details = []

        while remaining_names:
            best_info = None
            best_next = None

            # 贪心选择：在当前位置，找一个【路网距离】最近的下一个点
            for r in remaining_names:
                info = self.get_route_info(coords_map[current_name], coords_map[r])
                if info and (best_info is None or info['km'] < best_info['km']):
                    best_info = info
                    best_next = r

            if best_next:
                remaining_names.remove(best_next)
                route_details.append({"from": current_name, "to": best_next, **best_info})
                ordered_route.append(best_next)
                current_name = best_next
            else:
                break

        # 补齐无法解析的点
        for p in poi_list:
            if p not in ordered_route:
                ordered_route.append(p)

        return ordered_route, route_details