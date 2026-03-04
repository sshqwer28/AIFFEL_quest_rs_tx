
# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김태성
- 리뷰어 : 서인하


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - KITTI 데이터셋을 활용한 도로 Semantic Segmentation 전 과정(데이터 로드 → 전처리 → U-Net 학습 → U-Net++ 학습 → 결과 시각화 및 IoU 평가)이 오류 없이 완성되었습니다.
<img width="854" height="585" alt="image" src="https://github.com/user-attachments/assets/dc69a78b-415d-4b44-a262-aebe980d746b" />
    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
    - 핵심 코드 블록: KittiDataset 클래스 (Cell 5) 및 SkipPathways 클래스
    - KittiDataset을 가장 핵심적인 블록으로 판단한 이유는 KITTI 데이터의 입출력 구조를 정의하고 도로 레이블(== 7)을 이진 마스크로 변환하는 전처리 로직 전체를 담고 있기 때문입니다.
    - 클래스 수준과 메서드 수준 양쪽에 doc string이 체계적으로 달려 있어 각 파라미터의 의미를 명확히 이해할 수 있었습니다.
<img width="848" height="440" alt="image" src="https://github.com/user-attachments/assets/91fc4858-e895-43a4-b31d-781d5840ccfb" />

        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 추가 시도: U-Net++ + BCEDiceLoss 조합 실험
    - U-Net 학습 시에는 BCELoss를 사용했으나, U-Net++ 학습에서는 BCELoss와 DiceLoss를 결합한 BCEDiceLoss를 직접 구현하여 적용하는 추가 실험을 수행했습니다.
    - 도로 vs 비도로 클래스 불균형 문제에 DiceLoss가 더 강인하다는 점을 반영한 시도입니다.
<img width="822" height="285" alt="image" src="https://github.com/user-attachments/assets/ce4d2bbe-1568-4b54-b220-910e5d6df770" />

        
- [x]  **4. 회고를 잘 작성했나요?**
    - 정량 비교: U-Net IoU 0.7033 vs U-Net++ IoU 0.7308로 수치 비교 명시
    - 질적 분석: U-Net은 곡선 도로·소형 도로 영역 탐지 실패, U-Net++은 꺾인 도로와 작은 영역을 비교적 잘 masking한다는 시각적 분석 포함
    - 개선 방향: Boundary Loss 논문(arxiv:1812.07032)을 직접 인용하며 경계선 세그멘테이션 개선 방법을 구체적으로 제안
<img width="970" height="203" alt="image" src="https://github.com/user-attachments/assets/e13e3bba-dd74-435f-92d3-305294144682" />


- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 핵심 모듈화 사례: ConvLayer → ConvBlock → UpsamplingLayer → SkipPathways → UNetpp 계층적 설계
<img width="974" height="1033" alt="image" src="https://github.com/user-attachments/assets/82745c55-b9c9-43b0-b72c-c883d646022d" />


# 회고(참고 링크 및 코드 개선)
U-Net과 U-Net++을 모두 직접 구현하고, BCEDiceLoss라는 복합 손실 함수까지 실험한 점이 인상적이었습니다. 
특히 U-Net++의 nested skip connection 구조를 SkipPathways 모듈로 추상화한 설계가 가독성과 확장성 면에서 뛰어났습니다.
회고에서 Boundary Loss를 제안한 방향처럼 앙상블예측 등의 방향도 고려해볼 수 있을거 같습니다.

