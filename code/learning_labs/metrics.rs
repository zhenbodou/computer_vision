//! 附录 I：不需要第三方依赖的指标参考实现。
//! rustc --edition=2024 --test code/learning_labs/metrics.rs -o /tmp/cv_metrics_test
#[derive(Debug, Clone, Copy)]
struct Counts {
    tp: u32,
    fp: u32,
    fn_: u32,
    tn: u32,
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then(|| numerator / denominator)
}

impl Counts {
    fn precision(self) -> Option<f64> {
        ratio(f64::from(self.tp), f64::from(self.tp) + f64::from(self.fp))
    }
    fn recall(self) -> Option<f64> {
        ratio(f64::from(self.tp), f64::from(self.tp) + f64::from(self.fn_))
    }
    fn accuracy(self) -> Option<f64> {
        let correct = f64::from(self.tp) + f64::from(self.tn);
        ratio(correct, correct + f64::from(self.fp) + f64::from(self.fn_))
    }
    fn f1(self) -> Option<f64> {
        let twice_tp = 2.0 * f64::from(self.tp);
        ratio(
            twice_tp,
            twice_tp + f64::from(self.fp) + f64::from(self.fn_),
        )
    }
}

/// 零次错误、独立同分布二项试验的单侧置信上界。
/// alpha 是显著性水平，例如 0.05；没有样本时无法估计。
fn zero_error_upper(n: u32, alpha: f64) -> Option<f64> {
    if n == 0 || !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return None;
    }
    // 等价于 1 - alpha.powf(1/n)，exp_m1 避免两近似数相减损失精度。
    Some(-(alpha.ln() / f64::from(n)).exp_m1())
}

fn main() {
    let c = Counts {
        tp: 8,
        fp: 20,
        fn_: 2,
        tn: 970,
    };
    println!(
        "precision={:.6}, recall={:.6}, accuracy={:.6}, F1={:.6}",
        c.precision().unwrap(),
        c.recall().unwrap(),
        c.accuracy().unwrap(),
        c.f1().unwrap()
    );
    println!(
        "300 个独立样本零误报：单侧 95% 上界={:.6}",
        zero_error_upper(300, 0.05).unwrap()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(actual: Option<f64>, expected: f64) {
        assert!((actual.unwrap() - expected).abs() < 1e-10);
    }
    #[test]
    fn worked_example() {
        let c = Counts {
            tp: 8,
            fp: 20,
            fn_: 2,
            tn: 970,
        };
        close(c.precision(), 2.0 / 7.0);
        close(c.recall(), 0.8);
        close(c.accuracy(), 0.978);
        close(c.f1(), 8.0 / 19.0);
    }
    #[test]
    fn no_samples_is_undefined() {
        let c = Counts {
            tp: 0,
            fp: 0,
            fn_: 0,
            tn: 0,
        };
        assert!(c.precision().is_none());
        assert!(c.recall().is_none());
        assert!(c.accuracy().is_none());
        assert!(c.f1().is_none());
    }
    #[test]
    fn majority_classifier_misses_every_defect() {
        let c = Counts {
            tp: 0,
            fp: 0,
            fn_: 10,
            tn: 990,
        };
        close(c.accuracy(), 0.99);
        close(c.recall(), 0.0);
        close(c.f1(), 0.0);
        assert!(c.precision().is_none());
    }
    #[test]
    fn bound_and_required_sample_size() {
        close(
            zero_error_upper(300, 0.05),
            1.0 - 0.05_f64.powf(1.0 / 300.0),
        );
        assert!(zero_error_upper(2994, 0.05).unwrap() > 0.001);
        assert!(zero_error_upper(2995, 0.05).unwrap() <= 0.001);
        assert!(zero_error_upper(0, 0.05).is_none());
        for alpha in [0.0, 1.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(zero_error_upper(300, alpha).is_none());
        }
    }
    #[test]
    fn large_counts_do_not_overflow_integer_addition() {
        let c = Counts {
            tp: u32::MAX,
            fp: u32::MAX,
            fn_: u32::MAX,
            tn: u32::MAX,
        };
        close(c.precision(), 0.5);
        close(c.accuracy(), 0.5);
        close(c.f1(), 0.5);
    }
}
