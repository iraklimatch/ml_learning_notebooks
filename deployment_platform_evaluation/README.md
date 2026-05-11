# Deployment Platform Evaluation for an ML Model

Practice activity from the Microsoft Foundations of AI and Machine Learning course. The goal is to pick a real cloud platform to host an ML model in production, then back the choice up with actual criteria instead of vibes.

The three platforms I compared:

1. **Azure Machine Learning** (Microsoft)
2. **AWS SageMaker** (Amazon)
3. **Google Vertex AI** (Google Cloud)

All three are mature managed ML platforms. The question is which one fits best for the kind of project I would actually run, given that I work in a Microsoft-heavy environment and care about a clean integration story alongside the standard scale, cost, and ease of use stuff.

## Comparison Table

| Criteria | Azure Machine Learning | AWS SageMaker | Google Vertex AI |
|---|---|---|---|
| **Scalability** | Auto-scaling on AKS, GPU/CPU clusters, distributed training. Strong for both training and inference. | Best-in-class horizontal scaling. SageMaker endpoints autoscale by traffic, multi-model endpoints save cost. | Strong autoscaling, integrates tightly with GKE. Excellent for batch and online prediction. |
| **Cost** | Pay-as-you-go plus reserved instances (up to 60% off for 1-3 year commits). Free tier for compute instances under student/enterprise plans. | Granular pricing per training hour and per inference request. Spot instances cut training cost. Can get expensive fast without monitoring. | Competitive pricing, often slightly cheaper than AWS for similar workloads. Free credits for new accounts. |
| **Ease of use** | Studio UI is approachable for analysts, SDK v2 is clean for engineers. AutoML is a real productivity win. | Powerful but the surface area is huge. Steeper learning curve, more concepts (notebook instances, training jobs, endpoints, pipelines, etc.). | Notebook-first experience is solid. Newer than the other two so docs are still catching up in places. |
| **Integration with existing tools** | Native to Microsoft 365, Active Directory, Power BI, Azure DevOps, GitHub. Plugs into the rest of the Azure stack (Data Factory, Synapse, Databricks) with one click. | Plays best inside the AWS ecosystem (S3, Lambda, Step Functions, Redshift). Integration with non-AWS tools is fine but more work. | Best for teams already on Google Workspace, BigQuery, or GKE. Limited native ties to Microsoft tooling. |
| **Support for ML models** | First-class support for Scikit-learn, TensorFlow, PyTorch, ONNX. Built-in MLflow tracking, model registry, and managed endpoints. | Same framework support plus a long list of built-in algorithms. Strongest tooling for production MLOps (SageMaker Pipelines, Model Monitor, Feature Store). | Full framework support plus Vertex AI Pipelines and a polished AutoML for tabular, vision, and text. |
| **Performance** | Solid. GPU instances on demand, fast cold start times on managed endpoints. | Top-tier performance, especially for very high-throughput inference. Inference Recommender helps pick the right instance type. | Excellent performance backed by Google's infrastructure, especially for vision and NLP. |
| **Security & compliance** | FedRAMP, ISO 27001, SOC 2, GDPR, HIPAA, CJIS. Azure AD for identity, Key Vault for secrets, encryption everywhere. Strong for US public sector. | Very deep security tooling (IAM, VPC, KMS), strong compliance coverage. Standard for enterprise and government. | Strong compliance footprint, but smaller public sector presence in the US than Azure or AWS. |

## Analysis

The honest answer is that all three platforms could handle most ML deployment projects. The decision really comes down to which ecosystem the rest of the org already lives in and what tradeoffs you can stomach.

**Trade-offs I weighed:**

- AWS SageMaker has the deepest MLOps toolkit, but the surface area is overwhelming for a small team. The cost can also creep up fast because every piece is priced separately.
- Vertex AI is the slickest pure-ML experience and pricing is competitive, but the weak integration with Microsoft tooling is a problem for any org already running on Microsoft 365 and Active Directory.
- Azure ML is the most pragmatic middle ground. It is not the absolute best at any one thing, but it is strong across the board and the integration story with the rest of the Microsoft stack saves real engineering time.

**Critical limitations to watch:**

- **Azure** can lock you into Azure-specific services. Mitigate by storing data in open formats and using Kubernetes (AKS) for inference so workloads stay portable.
- **AWS** charges for everything, so even a small team can burn through budget if no one is watching. Cost guardrails are mandatory.
- **GCP** has a smaller pool of certified engineers in the labor market, so hiring and support can be harder.

## Justification

I picked **Azure Machine Learning** for this project. The reasoning:

1. **Integration wins more than raw features.** Most of the orgs I would deploy into already run Microsoft 365, Active Directory, Power BI, and GitHub. Azure ML plugs into that stack natively. Identity, single sign on, dashboards, and CI/CD all work out of the box. That cuts weeks of integration work compared to gluing AWS or GCP into a Microsoft shop.
2. **The pricing model fits a multi-year commitment.** Reserved instances cut compute cost by 30 to 60 percent for predictable workloads, which matters when the project is funded out of an annual budget rather than a startup runway. Spot VMs handle batch retraining cheaply.
3. **The tooling is good enough across the board.** AutoML, the model registry, MLflow tracking, managed online endpoints, and the SDK v2 all hit the marks I need. SageMaker has a deeper MLOps toolkit, but the difference does not pay back the integration tax for a Microsoft-centric org.
4. **Strong compliance coverage for public sector or regulated work.** FedRAMP, HIPAA, CJIS, and ISO 27001 are already in place, which matters if the project ever touches government, healthcare, or law enforcement data.

## Supporting Evidence

- Gartner's Magic Quadrant for Cloud AI Developer Services consistently places Azure ML, AWS SageMaker, and Vertex AI in the Leaders quadrant. Azure is usually called out for enterprise integration and hybrid cloud support.
- The Forrester Wave for AI/ML Platforms similarly ranks all three as Leaders. Azure scores particularly well on partner ecosystem and enterprise adoption.
- Microsoft case studies show Azure ML deployed at scale at organizations like ASOS (recommendations), Walgreens Boots Alliance (forecasting), and the City of Toledo (smart city). The pattern is clear: large orgs with existing Microsoft footprints pick Azure to avoid integration pain.
- On G2 and TrustRadius, Azure ML scores well on ease of use and customer support, slightly behind SageMaker on raw feature depth and slightly ahead of Vertex AI on enterprise readiness.

## Future Considerations

- **Scalability:** Azure ML on AKS scales horizontally as the model gets more traffic. Adding new regions or compute pools is straightforward.
- **Integration:** As the org adds new tools (Fabric, Synapse, Power BI dashboards), they plug in without extra connectors.
- **Cost efficiency:** Reserved instance commitments and spot VMs can be reviewed annually as workloads stabilize. Cost Management plus Budgets keeps spend visible.
- **MLOps maturity:** The integration with Azure DevOps and GitHub Actions makes it easy to grow from manual deployments to fully automated CI/CD pipelines without changing platforms.
- **Skill availability:** Microsoft certifications (DP-100, AI-102, AZ-900) are common in the labor market, so hiring or upskilling team members onto Azure is realistic.

## Conclusion

For a Microsoft-leaning organization with an annual budget, mixed technical skill across the team, and a need for strong compliance and integration, **Azure Machine Learning is the right pick**. AWS SageMaker would be the strongest alternative if the org were already AWS-native or needed the deepest possible MLOps tooling. Vertex AI is excellent technically but the weak Microsoft integration takes it out of contention for most enterprise scenarios I would actually face.

The bigger lesson from doing this comparison: the "best" platform is rarely the one with the most features. It is the one that fits the org's existing stack, budget cycle, and team skill level. Picking against that fit usually costs more in integration work and rework than it saves in any single platform advantage.
