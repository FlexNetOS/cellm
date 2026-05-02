// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! cellm-scheduler: Session state machine, queues, and thermal policy.

pub mod session;
pub mod queue;
pub mod thermal;
pub mod rr;
pub mod policy;
pub mod batch;

pub use session::Session;
pub use session::{SessionError, SessionState};
pub use queue::Queue;
pub use thermal::{ThermalLevel, ThermalPolicy};
pub use rr::{RoundRobinScheduler, SessionId};
pub use policy::{PolicyExecutor, SchedulingPlan, SchedulingPolicy};
pub use batch::{BatchDetector, BatchGroup, BatchSessionInfo, BatchedDecodeResult};
