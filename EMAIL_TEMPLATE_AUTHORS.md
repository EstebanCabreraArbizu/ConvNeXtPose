# Email Template for Authors

---

**Subject:** Request for Model L and M Checkpoints - ConvNeXtPose (IEEE Access 2023)

---

Dear ConvNeXtPose Authors,

I hope this email finds you well. I am writing regarding your excellent paper "ConvNeXtPose: A Fast Accurate Method for 3D Human Pose Estimation" published in IEEE Access 2023.

## Context

I am attempting to reproduce and evaluate the ConvNeXtPose-L and ConvNeXtPose-M models on the Human3.6M dataset (Protocol 2), as reported in your paper with impressive results:
- Model L: 42.3 mm MPJPE
- Model M: 44.6 mm MPJPE

## Issue Discovered

After downloading the checkpoints from the official Google Drive folder (ID: 12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI) linked in your README, I discovered that the checkpoint files appear to be **mislabeled**.

### Technical Analysis

I performed a detailed architecture inspection of the downloaded checkpoints:

```
File: ConvNeXtPose_L (1).tar
  - First layer shape: [48, 3, 4, 4]
  - Detected architecture: dims=[48, 96, 192, 384] → Model S
  - Total parameters: 8,391,354
  
File: ConvNeXtPose_M (1).tar
  - First layer shape: [48, 3, 4, 4]
  - Detected architecture: dims=[48, 96, 192, 384] → Model S
  - Total parameters: 7,596,986
```

**Expected architectures:**
- Model L should have: dims=[192, 384, 768, 1536] (first layer: [192, 3, 4, 4])
- Model M should have: dims=[64, 128, 256, 512] (first layer: [64, 3, 4, 4])

All three files (L, M, and S) appear to contain the Small model architecture, preventing reproduction of your reported results for the larger models.

## Request

Could you please:

1. **Verify** if the correct Model L and Model M checkpoints are available?
2. **Provide access** to the actual Model L (`dims=[192, 384, 768, 1536]`) and Model M (`dims=[64, 128, 256, 512]`) checkpoints?
3. **Update** the Google Drive folder with the correct files?

## Additional Information

I have successfully:
- Set up the complete testing pipeline on Kaggle
- Converted the PyTorch legacy checkpoint format to modern format
- Validated that Model S works correctly

I am ready to test models L and M as soon as the correct checkpoints are available.

For your reference, I have documented the complete analysis here:
- Detailed technical report: [attach CHECKPOINT_MISLABELING_ISSUE.md]
- Repository fork: https://github.com/EstebanCabreraArbizu/ConvNeXtPose

## Acknowledgment

Thank you for making your code publicly available and for your contribution to the 3D pose estimation field. Your work is highly valuable, and I am eager to validate and build upon your excellent results.

I look forward to your response.

Best regards,

---

**Alternative Contact Methods:**
- GitHub Issue: Create issue on the official repository
- Conference/Journal Contact: Through IEEE Access
- Research Gate / LinkedIn: Direct message to authors

---

## Attachments to Include:
1. CHECKPOINT_MISLABELING_ISSUE.md (detailed technical report)
2. Screenshots of analysis results (optional)
3. Error logs showing size mismatch (optional)
