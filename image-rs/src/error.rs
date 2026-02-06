use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

/// Return only the first line of a message, stripping any stack trace.
fn first_line(msg: &str) -> &str {
    msg.split('\n').next().unwrap_or(msg)
}

#[derive(Debug)]
pub enum Error {
    Candle(candle_core::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
    Msg(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Candle(e) => write!(f, "{}", first_line(&e.to_string())),
            Error::Io(e) => write!(f, "{}", first_line(&e.to_string())),
            Error::Json(e) => write!(f, "{}", first_line(&e.to_string())),
            Error::Msg(msg) => write!(f, "{}", first_line(msg)),
        }
    }
}

impl std::error::Error for Error {}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Candle(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::Msg(msg)
    }
}

impl From<&str> for Error {
    fn from(msg: &str) -> Self {
        Error::Msg(msg.to_string())
    }
}
