---
title: react-native-ble-plx 사용 방법
date: 2024-04-30
tags:
  - React
  - ReactNative
  - Beacon
  - BLE
  - Bluetooth
  - iBeacon
authors:
  - 이재희
slug: react-native-ble-plx-usage
---
# 개요
[[BLE 라이브러리]]에서 소개했던 라이브러리 중, `react-native-ble-plx`를 어떻게 사용하는지에 대해 기록을 남긴다.

# 패키지 설치
```bash
yarn add react-native-ble-plx
```
yarn 또는 npm을 사용하여 먼저 설치를 진행한다. 만약 `npx expo install react-native-ble-plx` 명령어를 사용하여 설치를 진행한다면, `app.json` 또는 `app.config.js`의 플러그인 부분에 라이브러리가 자동으로 추가되는 듯하다.

```bash
yarn expo prebuild
```
expo의 prebuild 명령어를 이용해 네이티브 파일을 생성한다.

```json
{
  "expo": {
    "plugins": ["react-native-ble-plx"]
  }
}
```
최상위 폴더의 app.json에 라이브러리를 추가하거나, 추가가 되어있는지 확인한다.

```bash
yarn android
```
안드로이드용으로 빌드 시 위 명령어를 입력한다. ios 빌드는 맥이 없어서 실행을 못 해봤다. ㅠ

[[BLE 라이브러리|BLE 라이브러리 정리]]에서도 설명했지만, `neverForLocation` 권한 플래그가 라이브러리 자체에서 포함됐기 때문에 `AndroidManifest.xml` 파일의 `BLUETOOTH_SCAN` 부분에 다음과 같이 `tools:remove` 속성을 추가해야 한다. 
```xml
 <uses-permission android:name="android.permission.BLUETOOTH_SCAN" tools:remove="android:usesPermissionFlags"/>
```

그리고 `AndroidManifest.xml` 최상단 `<manifest>` 태그에 다음과 같이 xmlns를 추가해줘야 빌드 시 오류가 나지 않는다.
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android" xmlns:tools="http://schemas.android.com/tools">
```


# 구현 코드

## 인스턴스 생성

다음과 같이 먼저 `BleManager` 인스턴스를 생성한다.
```ts
import { BleManager } from 'react-native-ble-plx'

export const manager = new BleManager()
```
반드시 **하나의 인스턴스만 허용**된다고 한다. 전역변수로 선언하면 되는 듯하다.
인스턴스를 삭제하려면 `manager.destroy()` 함수를 활용하면 된다. 

## 권한 부여 (Android)
```js
import { PermissionsAndroid, Platform, ToastAndroid } from "react-native";
import { BleManager, Device, ScanMode } from "react-native-ble-plx";

const requestBluetoothPermission = async () => {
  if (Platform.OS === "ios") {
    return true; // iOS는 권한 부여 코드가 따로 필요하지 않음
  }
  if (
    Platform.OS === "android" &&
    PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION
  ) {
    const apiLevel = parseInt(Platform.Version.toString(), 10);
    if (apiLevel < 31) {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION
      );
      return granted === PermissionsAndroid.RESULTS.GRANTED;
    }
    if (
      PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN &&
      PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT
    ) {
      const result = await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      ]);
      return (
        result["android.permission.BLUETOOTH_CONNECT"] ===
          PermissionsAndroid.RESULTS.GRANTED &&
        result["android.permission.BLUETOOTH_SCAN"] ===
          PermissionsAndroid.RESULTS.GRANTED &&
        result["android.permission.ACCESS_FINE_LOCATION"] ===
          PermissionsAndroid.RESULTS.GRANTED
      ); // 모든 권한이 허용되었으면 true를 반환
    }
    ToastAndroid.show("권한을 모두 허용해주세요.", 3000); // 권한 허용이 안 되어있으면 toast 메시지 출력
    return false;
  }
};
```

iOS는 따로 권한 요청 코드가 필요 없다. 

## 기기 검색
권한 요청이 성공하면 디바이스를 검색할 수 있다. 다음은 검색된 모든 기기의 이름을 출력하는 코드이다.
```ts
bleManager.startDeviceScan(null, {
	scanMode: ScanMode.Balanced
}, (error, device) => {
	if (error) {
		console.log(error);
		return error;
	}
	if (!device) return;
	console.log(device.name);
})
```
아까 생성한 `bleManager` 인스턴스의 `startDeviceScan` 함수로 스캔을 시작한다.
첫 번째 인자에 uuid를 입력하면 검색되는 기기를 해당 uuid에 대해 필터링할 수 있다. null을 입력하면 따로 필터링 과정을 거치지 않고 모든 기기가 검색된다.
인자로 넘긴 콜백 함수는 어떤 디바이스를 찾았을 때 호출된다.

# 끝

네이티브(Kotlin, Swift)에 비해 RN은 블루투스 등을 제어하기가 까다로운데, 이 라이브러리는 iOS와 안드로이드를 동시에 지원해서 생산성이 뛰어나다.
또한 코드 몇 줄로 백그라운드에서 실행이 가능한 것으로 보인다.

