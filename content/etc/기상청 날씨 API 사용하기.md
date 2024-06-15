---
title: 기상청 날씨 API 사용하기
date: 2024-06-10
tags:
  - API
  - 날씨
  - 기상청
  - Python
authors:
  - 이재희
slug: weather-api
---
# 개요
요즘 일교차가 심해 기온 그래프를 그려보려고 한다.
날씨 API 중 가장 많이 사용되는 OpenWeatherMap을 쓰려다가, 무료 계정은 단기 예보 API를 사용할 수 없다는 것을 깨닫고 불편함을 감수하며 기상청 API를 사용하기로 했다.

기상청 API는 상당히 불편하게 만들어놨다. 기본 리턴 값이 XML타입에다가, JSON 형식도 개떡같고 호출도 귀찮지만, 무료라서 쓸 이유는 충분하다고  생각한다.

# API 요청 날리기
기상청 API는 [공공데이터포털](https://www.data.go.kr)을 통해 제공된다. 귀찮지만 회원가입을 하고 API 신청을 해야 인증 토큰을 얻을 수 있다. 

그리고 다음 코드로 요청을 날렸는데.......
```python
now = datetime.now(KST)
base_date = now.strftime("%Y%m%d)
response = requests.get(
	"https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst",
	params={
		'serviceKey': API_KEY,
		'pageNo': 1,
		'numOfRows': 1000,
		'dataType': 'JSON',
		'base_date': base_date,
		'base_time': "0500",
		'nx': LAT,
		'ny': LNG,
	},
)
```

다음과 같은 응답이 날아왔다.
```json
{"response":{"header":{"resultCode":"01","resultMsg":"APPLICATION_ERROR"}}}
```
형식을 하라는대로 다 맞춰서 했는데 왜 안될까?

그렇게 몇 시간 동안 삽질한 결과, 기상청 API는 위도/경도를 사용하지 않고 자체적으로 나눈 행정 구역 단위를 쓴다고 한다. ㅋ

그래서 사용 설명서서를 다운받아 행정구역이 포함된 엑셀 파일로 내가 원하는 위치의 x, y 좌표값을 제대로 입력하니 해당 위치의 날씨를 받아올 수 있었다.

```json
{
  "response": {
    "header": {
      "resultCode": "00",
      "resultMsg": "NORMAL_SERVICE"
    },
    "body": {
      "dataType": "JSON",
      "items": {
        "item": [
          {
            "baseDate": "20240609",
            "baseTime": "0500",
            "category": "TMP",
            "fcstDate": "20240609",
            "fcstTime": "0600",
            "fcstValue": "17",
            "nx": 55,
            "ny": 127
          }
        ]
      },
      "pageNo": 1,
      "numOfRows": 1,
      "totalCount": 809
    }
  }
}
```

보기만 해도 머리가 아픈 구조이지만, 나는 포기하지 않고 데이터들을 정리해서 온도만 뽑아냈다.

그리고 추출한 온도 값만 활용하여 그래프를 그릴 수 있었다.

![](https://i.imgur.com/g2Mte4n.png)

굳이 콘솔로 그래프를 그린 이유는 멋져보여서다.

# 내가 배운 것
\
사용설명서를 읽자.
