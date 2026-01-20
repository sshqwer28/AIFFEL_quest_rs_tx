
# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김태성
- 리뷰어 : 서인하


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
	- 네, semantic segmentation 결과를 활용하여 인물모드 사진을 성공적으로 생성하였으며, 아웃포커싱 인물 사진, 동물 사진, 크로마키 배경 전환 사진이 각각 1장 이상 결과물로 첨부되어 문제에서 요구한 최종 결과를 충족하였습니다.
	- 근거
* ![인물 아웃포커싱 결과] (https://github.com/inhajourney/AIFFEL_quest_rs_tx-ook/blob/master/Exploration/Ex02/images/img1.png)
* ![동물 인물모드 결과] (https://github.com/inhajourney/AIFFEL_quest_rs_tx-ook/blob/master/Exploration/Ex02/images/img2.png)
* ![크로마키 배경 전환 결과] (https://github.com/inhajourney/AIFFEL_quest_rs_tx-ook/blob/master/Exploration/Ex02/images/img3.png)

    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
	- semantic segmentation mask를 후처리하여 경계 오류를 보완하는 로직이 본 프로젝트의 핵심이며, 해당 코드 블록에 각 처리 단계의 목적과 동작 원리를 설명하는 주석이 포함되어 있어 전체 흐름을 이해하는 데 도움이 되었습니다.
* segmentation mask 보정 로직
* 배경 블러 및 합성 처리 함수

        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
	- 인물 경계 부분에서 발생하는 mask 오류 문제를 명확히 인지하고, morphological operation 및 threshold 조정을 통해 이를 완화하려는 시도와 그 결과가 코드와 설명으로 잘 기록되어 있습니다.
* mask dilation/erosion 적용
* blur 강도 비교 실험
        
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
	- semantic segmentation 기반 인물모드의 한계점을 정리하고, 특히 머리카락·경계 영역에서의 오류 원인과 아쉬운 점을 명확히 서술하여 프로젝트를 통해 배운 점이 잘 드러났습니다.
* 인물모드의 구조적 한계 인식
* 데이터 품질과 segmentation 성능의 관계

        
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        -전체적으로 함수 단위로 잘 분리되어 있으며, 중복 코드를 최소화하려는 구조를 유지하고 있어 가독성과 재사용성 측면에서 적절한 수준의 코드 품질을 유지하고 있다고 판단됩니다.

# 회고(참고 링크 및 코드 개선)


