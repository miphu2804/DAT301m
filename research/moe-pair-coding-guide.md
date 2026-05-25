# MoE Pair-Coding Learning Guide

Muc tieu: tiep xuc MoE lan dau, hieu du truc giac de doc duoc DeepSeek paper, va tu code mot MoE mini trong `research/moe_test.ipynb`.

Nguon goc paper:
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- DeepSeek-V2: https://arxiv.org/abs/2405.04434
- DeepSeek-V3: https://arxiv.org/abs/2412.19437

## 1. Ban dang hoc cai gi

MoE = Mixture of Experts.

Truc giac ngan:
- Dense transformer: moi token di qua cung mot FFN lon.
- MoE transformer: moi token chi di qua mot vai "expert" nho.
- Ket qua: tong tham so tang manh, nhung compute moi token khong tang tuong ung.

Neu day la lan dau gap MoE, chi can nam 4 cau hoi:
1. Tai sao can MoE?
2. Token duoc gui toi expert nao?
3. Tai sao expert khong hoc trung nhau?
4. DeepSeek khac GShard/Switch o diem nao?

## 2. DeepSeekMoE noi gi, o muc de hieu

Paper `DeepSeekMoE` khong chi noi "dung MoE", ma de xuat 2 y chinh:

1. Cat expert thanh nhieu expert nho hon
- Paper mo ta y tuong "finely segmenting the experts into mN ones and activating mK from them".
- Truc giac: thay vi co it expert to, ta co nhieu expert nho.
- Loi ich: router co nhieu to hop expert hon de chon, nen de chuyen mon hoa hon.

2. Tach `shared experts`
- Paper mo ta viec "isolating Ks experts as shared ones".
- Truc giac: co mot it expert luon duoc dung de hoc kien thuc chung.
- Phan con lai la `routed experts`, duoc chon theo token.
- Loi ich: giam viec moi expert routed phai hoc lai kien thuc pho thong.

Tom lai:
- Dense FFN = 1 khoi kien thuc chung.
- MoE co ban = nhieu khoi, router chon top-k.
- DeepSeekMoE = nhieu expert nho hon + them shared experts.

## 3. DeepSeek-V2/V3 bo sung gi

Cho nguoi moi, khong can hoc het V2/V3 ngay. Chi can biet:

- `DeepSeek-V2` noi rang mo hinh dung `DeepSeekMoE` + `MLA`.
- `DeepSeek-V3` giu DeepSeekMoE, them `auxiliary-loss-free` load balancing.

Cai nay quan trong vi:
- `DeepSeekMoE` giai bai toan kien truc expert.
- `V2/V3` giai bai toan scale he thong va training efficiency.

Neu moi hoc, uu tien thu tu:
1. Hieu top-k routing
2. Hieu vi sao expert bi collapse / mat can bang
3. Hieu shared expert cua DeepSeek
4. Sau do moi doc load balancing cua V3

## 4. Mental model toi muon ban giu trong dau

Tuong tuong:
- Router la le tan.
- Experts la nhieu bac si chuyen khoa.
- Shared experts la bac si tong quat, ai cung gap.
- Routed experts la bac si chuyen khoa, chi gap khi phu hop.

Neu khong co shared experts:
- Moi expert chuyen khoa vua phai hoc kien thuc chung, vua phai hoc kien thuc rieng.
- Rat de bi lap thong tin.

Neu co shared experts:
- Kien thuc pho thong di vao shared experts.
- Routed experts co xu huong hoc pattern dac thu hon.

## 5. Pair-coding roadmap cho buoi dau

Thoi luong goi y: 90-120 phut.

### Phase 1: Dung truc giac, chua doc cong thuc

Muc tieu:
- Biet FFN dense hoat dong the nao
- Biet router trong MoE lam gi

Viec can lam trong notebook:
1. Tao tensor input `x` shape `(batch, seq, dModel)`.
2. Viet 1 `DenseFFN` nho: `Linear -> GELU/ReLU -> Linear`.
3. In shape input/output.
4. Tu hoi: "neu 8 expert ma moi token chi dung 2 expert, compute thay doi the nao?"

Checkpoint:
- Ban giai thich duoc "MoE khong phai ensemble final output, ma la sparse activation trong layer".

### Phase 2: Viet MoE mini chua co shared expert

Muc tieu:
- Hieu top-k routing bang code.

Viec can code:
1. Tao `nExperts` FFN doc lap.
2. Tao `router = Linear(dModel, nExperts)`.
3. Tinh `routerLogits`.
4. Lay `topk`.
5. Chi chay nhung expert duoc chon.
6. Tron output expert bang trong so router.

Pseudo:

```python
routerLogits = router(x)
routerProb = softmax(routerLogits, dim=-1)
topkProb, topkIdx = torch.topk(routerProb, k=2, dim=-1)
```

Checkpoint:
- Ban biet vi sao "tong model to hon" nhung "moi token khong chay qua tat ca expert".

### Phase 3: Them thong ke de nhin thay imbalance

Muc tieu:
- Nhin ra van de mot vai expert bi dung qua nhieu, expert khac bi bo doi.

Viec can code:
1. Dem so token moi expert nhan.
2. Ve bar chart phan bo token/expert.
3. Thu input ngau nhien va input co pattern.

Ban can nhan ra:
- Router khong tu dong can bang.
- Chuyen mon hoa va load balancing la 2 viec lien quan nhung khong giong nhau.

### Phase 4: Them `shared experts` kieu DeepSeek

Muc tieu:
- Cam duoc cai moi cua DeepSeekMoE.

Viec can code:
1. Tao 1-2 `sharedExperts`.
2. Shared experts luon chay voi moi token.
3. Routed experts van top-k nhu cu.
4. Cong output cua shared + routed.

Checkpoint:
- Ban giai thich duoc tai sao shared experts giup routed experts do phai hoc kien thuc chung.

### Phase 5: Doc lai abstract cua DeepSeek

Sau khi code xong Phase 4, doc lai abstract `DeepSeekMoE`.

Luc nay 2 cau se "click":
- finely segmenting experts
- isolating shared experts

Neu doc paper truoc khi code, 2 cum nay rat de troi qua mat.

## 6. Scaffold code toi de xuat

Ban co the dung scaffold nay trong notebook:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, dModel, dHidden):
        super().__init__()
        self.fc1 = nn.Linear(dModel, dHidden)
        self.fc2 = nn.Linear(dHidden, dModel)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SimpleMoE(nn.Module):
    def __init__(self, dModel=16, dHidden=32, nExperts=4, topK=2):
        super().__init__()
        self.topK = topK
        self.experts = nn.ModuleList(
            [Expert(dModel, dHidden) for _ in range(nExperts)]
        )
        self.router = nn.Linear(dModel, nExperts)

    def forward(self, x):
        routerLogits = self.router(x)
        routerProb = F.softmax(routerLogits, dim=-1)
        topkProb, topkIdx = torch.topk(routerProb, k=self.topK, dim=-1)

        out = torch.zeros_like(x)

        for expertIdx, expert in enumerate(self.experts):
            mask = topkIdx == expertIdx
            if not mask.any():
                continue

            tokenMask = mask.any(dim=-1)
            selectedX = x[tokenMask]
            expertOut = expert(selectedX)

            gate = topkProb[mask].unsqueeze(-1)
            out[tokenMask] += gate * expertOut

        return out, routerProb, topkIdx
```

Luu y:
- Day la scaffold hoc tap, chua toi uu.
- Muc dich la hieu routing, khong phai viet kernel dep.
- Neu co bug shape, dung no de hoc; dung nhay thang toi implementation production.

## 7. Cac cau hoi pair-coding toi se hoi ban

Trong luc code, tu dung lai o moi diem nay:

1. Router dang chon expert dua tren gi?
2. Tai sao phai `softmax` truoc `topk`?
3. Neu expert A duoc chon cho gan het token thi dieu gi xay ra?
4. Shared expert khac residual block thong thuong o dau?
5. "Nhieu expert nho" co loi gi hon "it expert to"?

Neu ban tra loi duoc 5 cau nay bang loi cua ban, nen tang da chac.

## 8. Thu tu doc paper

Thu tu toi de xuat:

1. `DeepSeekMoE` abstract + introduction
2. Quay lai notebook, code top-k routing
3. `DeepSeekMoE` phan noi ve fine-grained experts va shared experts
4. Quay lai notebook, them shared experts
5. `DeepSeek-V2` abstract de thay MoE duoc dua vao model lon the nao
6. `DeepSeek-V3` abstract de thay bai toan load balancing duoc day xa hon

Khong can doc full V2/V3 ngay buoi dau.

## 9. Nhan biet 3 hieu nham pho bien

Hieu nham 1:
- "MoE = ensemble"
- Sai mot nua. MoE co nhieu subnet, nhung routing xay ra ben trong layer va sparse theo token.

Hieu nham 2:
- "Tong parameter lon hon => compute chac chan lon hon"
- Sai. MoE tang total params, nhung activated params moi token co the nho hon rat nhieu.

Hieu nham 3:
- "Chi can top-k router la xong"
- Sai. Van de that su la specialization, collapse, communication cost, load balancing.

## 10. Binh dan hoa DeepSeek bang 1 cau

DeepSeekMoE co the duoc nho nhu sau:

"Thay vi it chuyen gia lon, tao nhieu chuyen gia nho hon, goi them mot vai chuyen gia tong quat luon online, roi de router phoi hop."

## 11. Bai tap tiep theo sau buoi dau

Neu buoi dau on roi, buoi tiep theo lam 3 viec:

1. Them `capacity` gioi han so token moi expert.
2. Them auxiliary load-balancing loss ban don gian.
3. So sanh `DenseFFN` vs `SimpleMoE` tren toy task nho.

## 12. Cach toi de nghi minh pair-code tiep

Neu ban muon, buoi tiep theo toi co the di theo 1 trong 3 huong:

1. `Notebook-first`
- Toi se huong dan tung cell de dien vao `research/moe_test.ipynb`.

2. `Math-first`
- Toi se giai thich cong thuc gating, top-k, va vi sao MoE tiet kiem FLOPs.

3. `Paper-first`
- Toi se doc cung ban `DeepSeekMoE` section by section, moi section doi ra code.

## Unresolved questions

- Ban muon guide nghieng ve `PyTorch implementation` hay `paper reading` nhieu hon?
- Ban muon toi sua truc tiep `research/moe_test.ipynb` thanh notebook hoc tung cell khong?
