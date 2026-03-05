module 0x0::SecureIoTMonitoring {
    use iota::object::{Self, UID};
    use iota::tx_context::{Self, TxContext};
    use iota::transfer;
    use iota::event;
    use iota::clock::{Self, Clock};
    use iota::table::{Self, Table};
    use std::string::{Self, String};
    use std::vector;

    // ============= ERROR CODES =============
    const E_ARRAY_LENGTH_MISMATCH: u64 = 1;
    const E_NO_READINGS: u64 = 2;
    const E_NO_ANOMALIES: u64 = 3;
    const E_INVALID_SIGNATURE: u64 = 4;
    const E_INVALID_ENCRYPTION: u64 = 5;
    const E_NOT_AUTHORIZED: u64 = 6;
    const E_INVALID_STATUS: u64 = 7;

    // ============= ENCRYPTION METADATA =============
    
    /// Encryption configuration
    public struct EncryptionConfig has store, copy, drop {
        algorithm: String,          // "AES-256-GCM", "ChaCha20-Poly1305"
        key_derivation: String,     // "PBKDF2", "Argon2", "HKDF"
        iv_length: u64,            // Initialization vector length in bytes
        tag_length: u64,           // Authentication tag length in bytes
        version: String            // "v1.0"
    }

    // ============= SENSOR DATA STRUCTURES =============

    /// Plaintext sensor reading
    public struct SensorReading has store, drop, copy {
        reading_id: u64,
        temperature: u64,          // Fixed point (*100): 21.8°C = 2180
        humidity: u64,             // Fixed point (*100): 51.1% = 5110
        timestamp: u64,            // Unix timestamp
        is_negative_temp: bool,
        is_negative_humidity: bool,
        device_id: String,         // Device identifier
        location: String,          // Optional location info
        status: String             // "normal", "warning", "critical"
    }

    /// Encrypted sensor reading
    public struct EncryptedReading has store, drop, copy {
        reading_id: u64,
        encrypted_data: vector<u8>,     // Encrypted payload (temp + humidity + timestamp)
        encryption_iv: vector<u8>,      // Initialization vector
        auth_tag: vector<u8>,           // Authentication tag (for GCM/Poly1305)
        encryption_config: EncryptionConfig,
        timestamp: u64,                 // Plaintext timestamp (for indexing)
        device_id: String,              // Plaintext device ID (for querying)
        
        // TROPIC01 signature on encrypted data
        encrypted_data_hash: vector<u8>,
        signature: vector<u8>,
        public_key: vector<u8>
    }

    // ============= ANOMALY STRUCTURES =============

    /// Anomaly status tracking
    public struct AnomalyStatus has store, drop, copy {
        severity: String,           // "low", "medium", "high", "critical"
        confidence: u64,            // Confidence score (0-10000)
        is_confirmed: bool,         // Manual confirmation flag
        is_false_positive: bool,    // Marked as false positive
        resolution_notes: String,   // Resolution or explanation
        confirmed_by: address,      // Who confirmed/resolved
        confirmed_at: u64          // When confirmed
    }

    /// Plaintext anomaly record
    public struct AnomalyRecord has store, drop, copy {
        anomaly_id: u64,
        reading_id: u64,           // Links to original reading
        
        // Sensor values
        temperature: u64,
        humidity: u64,
        timestamp: u64,
        
        // Anomaly detection data
        anomaly_score: u64,        // Fixed point (*10000): -0.1234 = 1234
        is_anomaly: bool,
        anomaly_type: String,      // "temperature", "humidity", "both", "pattern"
        deviation_percent: u64,    // How much it deviates from normal (*100)
        
        // Model information
        model_version: String,
        detection_timestamp: u64,
        
        // TROPIC01 signature
        data_hash: vector<u8>,
        signature: vector<u8>,
        public_key: vector<u8>,
        
        // Status tracking
        status: AnomalyStatus
    }

    /// Encrypted anomaly record (for sensitive data)
    public struct EncryptedAnomalyRecord has store, drop, copy {
        anomaly_id: u64,
        reading_id: u64,
        
        // Encrypted anomaly details
        encrypted_data: vector<u8>,     // Contains: temp, humidity, score, type
        encryption_iv: vector<u8>,
        auth_tag: vector<u8>,
        encryption_config: EncryptionConfig,
        
        // Plaintext metadata (for querying)
        timestamp: u64,
        is_anomaly: bool,
        device_id: String,
        
        // TROPIC01 signature on encrypted data
        encrypted_data_hash: vector<u8>,
        signature: vector<u8>,
        public_key: vector<u8>,
        
        // Status tracking
        status: AnomalyStatus
    }

    // ============= STORAGE STRUCTURE =============

    /// Main storage with dual mode support
    public struct SensorStorage has key {
        id: UID,
        
        // Device management
        registered_devices: Table<address, String>,     // address -> device_id
        device_metadata: Table<String, DeviceMetadata>,
        
        // Plaintext storage
        device_readings: Table<address, vector<SensorReading>>,
        device_anomalies: Table<address, vector<AnomalyRecord>>,
        
        // Encrypted storage
        encrypted_readings: Table<address, vector<EncryptedReading>>,
        encrypted_anomalies: Table<address, vector<EncryptedAnomalyRecord>>,
        
        // Statistics and aggregates (always plaintext for analytics)
        daily_stats: Table<vector<u8>, DailyStatistics>,  // date_hash -> stats
        anomaly_summary: Table<address, AnomalySummary>,
        
        // Counters
        reading_counter: Table<address, u64>,
        anomaly_counter: Table<address, u64>,
        
        // Configuration
        default_encryption_config: EncryptionConfig,
        active_model_version: String,
        total_anomalies_detected: u64,
        
        // Access control
        authorized_viewers: Table<address, vector<address>>,  // owner -> list of viewers
        
        // CSV hashes (for data integrity verification)
        csv_hashes: Table<vector<u8>, u64>
    }

    /// Device metadata
    public struct DeviceMetadata has store, drop, copy {
        device_id: String,
        device_name: String,
        location: String,
        installation_date: u64,
        last_reading: u64,
        total_readings: u64,
        encryption_enabled: bool,
        public_key: vector<u8>     // TROPIC01 public key
    }

    /// Daily statistics (always plaintext for analytics)
    public struct DailyStatistics has store, drop, copy {
        date: String,              // "2026-02-15"
        device: address,
        
        // Temperature stats
        temp_min: u64,
        temp_max: u64,
        temp_avg: u64,
        temp_readings: u64,
        
        // Humidity stats
        humidity_min: u64,
        humidity_max: u64,
        humidity_avg: u64,
        humidity_readings: u64,
        
        // Anomaly stats
        anomalies_detected: u64,
        critical_anomalies: u64,
        false_positives: u64
    }

    /// Anomaly summary per device
    public struct AnomalySummary has store, drop, copy {
        device: address,
        total_anomalies: u64,
        unresolved_anomalies: u64,
        critical_anomalies: u64,
        false_positives: u64,
        last_anomaly_time: u64,
        anomaly_rate: u64          // Per 1000 readings (*1000)
    }

    // ============= EVENTS =============

    public struct ReadingStored has copy, drop {
        device: address,
        reading_id: u64,
        timestamp: u64,
        is_encrypted: bool,
        temperature: u64,          // 0 if encrypted
        humidity: u64              // 0 if encrypted
    }

    public struct AnomalyDetected has copy, drop {
        device: address,
        anomaly_id: u64,
        reading_id: u64,
        is_anomaly: bool,
        severity: String,
        is_encrypted: bool,
        timestamp: u64
    }

    public struct AnomalyStatusUpdated has copy, drop {
        device: address,
        anomaly_id: u64,
        new_status: String,
        updated_by: address,
        timestamp: u64
    }

    public struct DeviceRegistered has copy, drop {
        device: address,
        device_id: String,
        encryption_enabled: bool,
        timestamp: u64
    }

    public struct AccessGranted has copy, drop {
        owner: address,
        viewer: address,
        timestamp: u64
    }

    // ============= INITIALIZATION =============

    fun init(ctx: &mut TxContext) {
        let default_config = EncryptionConfig {
            algorithm: string::utf8(b"AES-256-GCM"),
            key_derivation: string::utf8(b"PBKDF2"),
            iv_length: 12,
            tag_length: 16,
            version: string::utf8(b"v1.0")
        };

        let storage = SensorStorage {
            id: object::new(ctx),
            registered_devices: table::new(ctx),
            device_metadata: table::new(ctx),
            device_readings: table::new(ctx),
            device_anomalies: table::new(ctx),
            encrypted_readings: table::new(ctx),
            encrypted_anomalies: table::new(ctx),
            daily_stats: table::new(ctx),
            anomaly_summary: table::new(ctx),
            reading_counter: table::new(ctx),
            anomaly_counter: table::new(ctx),
            default_encryption_config: default_config,
            active_model_version: string::utf8(b"isolation_forest_v1.0"),
            total_anomalies_detected: 0,
            authorized_viewers: table::new(ctx),
            csv_hashes: table::new(ctx)
        };
        
        transfer::share_object(storage);
    }

    // ============= DEVICE REGISTRATION =============

    /// Register a new device
    public entry fun register_device(
        storage: &mut SensorStorage,
        device_id: vector<u8>,
        device_name: vector<u8>,
        location: vector<u8>,
        public_key: vector<u8>,
        enable_encryption: bool,
        clock: &Clock,
        ctx: &mut TxContext
    ) {
        let sender = tx_context::sender(ctx);
        let timestamp = clock::timestamp_ms(clock) / 1000;
        
        assert!(vector::length(&public_key) == 64, E_INVALID_SIGNATURE);
        
        let device_id_str = string::utf8(device_id);
        
        let metadata = DeviceMetadata {
            device_id: device_id_str,
            device_name: string::utf8(device_name),
            location: string::utf8(location),
            installation_date: timestamp,
            last_reading: 0,
            total_readings: 0,
            encryption_enabled: enable_encryption,
            public_key
        };
        
        table::add(&mut storage.registered_devices, sender, device_id_str);
        table::add(&mut storage.device_metadata, device_id_str, metadata);
        table::add(&mut storage.reading_counter, sender, 0);
        table::add(&mut storage.anomaly_counter, sender, 0);
        
        // Initialize storage tables
        table::add(&mut storage.device_readings, sender, vector::empty());
        table::add(&mut storage.encrypted_readings, sender, vector::empty());
        table::add(&mut storage.device_anomalies, sender, vector::empty());
        table::add(&mut storage.encrypted_anomalies, sender, vector::empty());
        
        event::emit(DeviceRegistered {
            device: sender,
            device_id: device_id_str,
            encryption_enabled: enable_encryption,
            timestamp
        });
    }

    // ============= PLAINTEXT READING STORAGE =============

    /// Store plaintext sensor reading
    public entry fun store_reading_plaintext(
        storage: &mut SensorStorage,
        temperature: u64,
        humidity: u64,
        is_negative_temp: bool,
        is_negative_humidity: bool,
        location: vector<u8>,
        status: vector<u8>,
        data_hash: vector<u8>,
        signature: vector<u8>,
        public_key: vector<u8>,
        clock: &Clock,
        ctx: &mut TxContext
    ) {
        let sender = tx_context::sender(ctx);
        let timestamp = clock::timestamp_ms(clock) / 1000;
        
        // Validate signature
        assert!(vector::length(&signature) > 0, E_INVALID_SIGNATURE);
        assert!(vector::length(&public_key) == 64, E_INVALID_SIGNATURE);
        assert!(vector::length(&data_hash) == 32, E_INVALID_SIGNATURE);
        
        // Get device ID
        let device_id = if (table::contains(&storage.registered_devices, sender)) {
            *table::borrow(&storage.registered_devices, sender)
        } else {
            string::utf8(b"unknown")
        };
        
        // Get reading ID
        let counter = table::borrow_mut(&mut storage.reading_counter, sender);
        let reading_id = *counter;
        *counter = *counter + 1;
        
        let reading = SensorReading {
            reading_id,
            temperature,
            humidity,
            timestamp,
            is_negative_temp,
            is_negative_humidity,
            device_id,
            location: string::utf8(location),
            status: string::utf8(status)
        };
        
        let readings = table::borrow_mut(&mut storage.device_readings, sender);
        vector::push_back(readings, reading);
        
        // Update device metadata
        if (table::contains(&storage.device_metadata, device_id)) {
            let metadata = table::borrow_mut(&mut storage.device_metadata, device_id);
            metadata.last_reading = timestamp;
            metadata.total_readings = metadata.total_readings + 1;
        };
        
        event::emit(ReadingStored {
            device: sender,
            reading_id,
            timestamp,
            is_encrypted: false,
            temperature,
            humidity
        });
    }

    // ============= ENCRYPTED READING STORAGE =============

    /// Store encrypted sensor reading
    public entry fun store_reading_encrypted(
        storage: &mut SensorStorage,
        encrypted_data: vector<u8>,
        encryption_iv: vector<u8>,
        auth_tag: vector<u8>,
        algorithm: vector<u8>,
        key_derivation: vector<u8>,
        encrypted_data_hash: vector<u8>,
        signature: vector<u8>,
        public_key: vector<u8>,
        clock: &Clock,
        ctx: &mut TxContext
    ) {
        let sender = tx_context::sender(ctx);
        let timestamp = clock::timestamp_ms(clock) / 1000;
        
        // Validate encryption and signature
        assert!(vector::length(&encrypted_data) > 0, E_INVALID_ENCRYPTION);
        assert!(vector::length(&encryption_iv) > 0, E_INVALID_ENCRYPTION);
        assert!(vector::length(&auth_tag) > 0, E_INVALID_ENCRYPTION);
        assert!(vector::length(&signature) > 0, E_INVALID_SIGNATURE);
        assert!(vector::length(&public_key) == 64, E_INVALID_SIGNATURE);
        assert!(vector::length(&encrypted_data_hash) == 32, E_INVALID_SIGNATURE);
        
        let device_id = if (table::contains(&storage.registered_devices, sender)) {
            *table::borrow(&storage.registered_devices, sender)
        } else {
            string::utf8(b"unknown")
        };
        
        let counter = table::borrow_mut(&mut storage.reading_counter, sender);
        let reading_id = *counter;
        *counter = *counter + 1;
        
        let config = EncryptionConfig {
            algorithm: string::utf8(algorithm),
            key_derivation: string::utf8(key_derivation),
            iv_length: vector::length(&encryption_iv),
            tag_length: vector::length(&auth_tag),
            version: string::utf8(b"v1.0")
        };
        
        let reading = EncryptedReading {
            reading_id,
            encrypted_data,
            encryption_iv,
            auth_tag,
            encryption_config: config,
            timestamp,
            device_id,
            encrypted_data_hash,
            signature,
            public_key
        };
        
        let readings = table::borrow_mut(&mut storage.encrypted_readings, sender);
        vector::push_back(readings, reading);
        
        event::emit(ReadingStored {
            device: sender,
            reading_id,
            timestamp,
            is_encrypted: true,
            temperature: 0,
            humidity: 0
        });
    }

    // ============= PLAINTEXT ANOMALY RECORDING =============

    /// Record plaintext anomaly
    public entry fun record_anomaly_plaintext(
        storage: &mut SensorStorage,
        reading_id: u64,
        temperature: u64,
        humidity: u64,
        timestamp: u64,
        anomaly_score: u64,
        is_anomaly: bool,
        anomaly_type: vector<u8>,
        deviation_percent: u64,
        severity: vector<u8>,
        confidence: u64,
        data_hash: vector<u8>,
        signature: vector<u8>,
        public_key: vector<u8>,
        model_version: vector<u8>,
        clock: &Clock,
        ctx: &mut TxContext
    ) {
        let sender = tx_context::sender(ctx);
        let detection_timestamp = clock::timestamp_ms(clock) / 1000;
        
        // Validate
        assert!(vector::length(&signature) > 0, E_INVALID_SIGNATURE);
        assert!(vector::length(&public_key) == 64, E_INVALID_SIGNATURE);
        assert!(vector::length(&data_hash) == 32, E_INVALID_SIGNATURE);
        
        let counter = table::borrow_mut(&mut storage.anomaly_counter, sender);
        let anomaly_id = *counter;
        *counter = *counter + 1;
        
        let status = AnomalyStatus {
            severity: string::utf8(severity),
            confidence,
            is_confirmed: false,
            is_false_positive: false,
            resolution_notes: string::utf8(b""),
            confirmed_by: sender,
            confirmed_at: 0
        };
        
        let anomaly = AnomalyRecord {
            anomaly_id,
            reading_id,
            temperature,
            humidity,
            timestamp,
            anomaly_score,
            is_anomaly,
            anomaly_type: string::utf8(anomaly_type),
            deviation_percent,
            model_version: string::utf8(model_version),
            detection_timestamp,
            data_hash,
            signature,
            public_key,
            status
        };
        
        let anomalies = table::borrow_mut(&mut storage.device_anomalies, sender);
        vector::push_back(anomalies, anomaly);
        
        if (is_anomaly) {
            storage.total_anomalies_detected = storage.total_anomalies_detected + 1;
        };
        
        event::emit(AnomalyDetected {
            device: sender,
            anomaly_id,
            reading_id,
            is_anomaly,
            severity: string::utf8(severity),
            is_encrypted: false,
            timestamp: detection_timestamp
        });
    }

    // ============= ENCRYPTED ANOMALY RECORDING =============

    /// Record encrypted anomaly
    public entry fun record_anomaly_encrypted(
        storage: &mut SensorStorage,
        reading_id: u64,
        encrypted_data: vector<u8>,
        encryption_iv: vector<u8>,
        auth_tag: vector<u8>,
        timestamp: u64,
        is_anomaly: bool,
        severity: vector<u8>,
        confidence: u64,
        encrypted_data_hash: vector<u8>,
        signature: vector<u8>,
        public_key: vector<u8>,
        clock: &Clock,
        ctx: &mut TxContext
    ) {
        let sender = tx_context::sender(ctx);
        let detection_timestamp = clock::timestamp_ms(clock) / 1000;
        
        assert!(vector::length(&encrypted_data) > 0, E_INVALID_ENCRYPTION);
        assert!(vector::length(&signature) > 0, E_INVALID_SIGNATURE);
        
        let device_id = if (table::contains(&storage.registered_devices, sender)) {
            *table::borrow(&storage.registered_devices, sender)
        } else {
            string::utf8(b"unknown")
        };
        
        let counter = table::borrow_mut(&mut storage.anomaly_counter, sender);
        let anomaly_id = *counter;
        *counter = *counter + 1;
        
        let status = AnomalyStatus {
            severity: string::utf8(severity),
            confidence,
            is_confirmed: false,
            is_false_positive: false,
            resolution_notes: string::utf8(b""),
            confirmed_by: sender,
            confirmed_at: 0
        };
        
        let anomaly = EncryptedAnomalyRecord {
            anomaly_id,
            reading_id,
            encrypted_data,
            encryption_iv,
            auth_tag,
            encryption_config: storage.default_encryption_config,
            timestamp,
            is_anomaly,
            device_id,
            encrypted_data_hash,
            signature,
            public_key,
            status
        };
        
        let anomalies = table::borrow_mut(&mut storage.encrypted_anomalies, sender);
        vector::push_back(anomalies, anomaly);
        
        if (is_anomaly) {
            storage.total_anomalies_detected = storage.total_anomalies_detected + 1;
        };
        
        event::emit(AnomalyDetected {
            device: sender,
            anomaly_id,
            reading_id,
            is_anomaly,
            severity: string::utf8(severity),
            is_encrypted: true,
            timestamp: detection_timestamp
        });
    }

    // ============= STATUS MANAGEMENT =============

    /// Update anomaly status (plaintext)
    public entry fun update_anomaly_status_plaintext(
        storage: &mut SensorStorage,
        anomaly_id: u64,
        is_confirmed: bool,
        is_false_positive: bool,
        resolution_notes: vector<u8>,
        clock: &Clock,
        ctx: &mut TxContext
    ) {
        let sender = tx_context::sender(ctx);
        let timestamp = clock::timestamp_ms(clock) / 1000;
        
        assert!(table::contains(&storage.device_anomalies, sender), E_NO_ANOMALIES);
        
        let anomalies = table::borrow_mut(&mut storage.device_anomalies, sender);
        let len = vector::length(anomalies);
        
        let mut i = 0;
        while (i < len) {
            let anomaly = vector::borrow_mut(anomalies, i);
            if (anomaly.anomaly_id == anomaly_id) {
                anomaly.status.is_confirmed = is_confirmed;
                anomaly.status.is_false_positive = is_false_positive;
                anomaly.status.resolution_notes = string::utf8(resolution_notes);
                anomaly.status.confirmed_by = sender;
                anomaly.status.confirmed_at = timestamp;
                
                event::emit(AnomalyStatusUpdated {
                    device: sender,
                    anomaly_id,
                    new_status: if (is_false_positive) {
                        string::utf8(b"false_positive")
                    } else if (is_confirmed) {
                        string::utf8(b"confirmed")
                    } else {
                        string::utf8(b"pending")
                    },
                    updated_by: sender,
                    timestamp
                });
                
                break
            };
            i = i + 1;
        };
    }

    /// Update encrypted anomaly status
    public entry fun update_anomaly_status_encrypted(
        storage: &mut SensorStorage,
        anomaly_id: u64,
        is_confirmed: bool,
        is_false_positive: bool,
        resolution_notes: vector<u8>,
        clock: &Clock,
        ctx: &mut TxContext
    ) {
        let sender = tx_context::sender(ctx);
        let timestamp = clock::timestamp_ms(clock) / 1000;
        
        assert!(table::contains(&storage.encrypted_anomalies, sender), E_NO_ANOMALIES);
        
        let anomalies = table::borrow_mut(&mut storage.encrypted_anomalies, sender);
        let len = vector::length(anomalies);
        
        let mut i = 0;
        while (i < len) {
            let anomaly = vector::borrow_mut(anomalies, i);
            if (anomaly.anomaly_id == anomaly_id) {
                anomaly.status.is_confirmed = is_confirmed;
                anomaly.status.is_false_positive = is_false_positive;
                anomaly.status.resolution_notes = string::utf8(resolution_notes);
                anomaly.status.confirmed_by = sender;
                anomaly.status.confirmed_at = timestamp;
                
                event::emit(AnomalyStatusUpdated {
                    device: sender,
                    anomaly_id,
                    new_status: string::utf8(b"updated"),
                    updated_by: sender,
                    timestamp
                });
                
                break
            };
            i = i + 1;
        };
    }

    // ============= ACCESS CONTROL =============

    /// Grant viewing access to another address
    public entry fun grant_access(
        storage: &mut SensorStorage,
        viewer: address,
        clock: &Clock,
        ctx: &mut TxContext
    ) {
        let sender = tx_context::sender(ctx);
        let timestamp = clock::timestamp_ms(clock) / 1000;
        
        if (!table::contains(&storage.authorized_viewers, sender)) {
            table::add(&mut storage.authorized_viewers, sender, vector::empty());
        };
        
        let viewers = table::borrow_mut(&mut storage.authorized_viewers, sender);
        vector::push_back(viewers, viewer);
        
        event::emit(AccessGranted {
            owner: sender,
            viewer,
            timestamp
        });
    }

    // ============= QUERY FUNCTIONS =============

    /// Get plaintext reading count
    public fun get_plaintext_reading_count(
        storage: &SensorStorage,
        device: address
    ): u64 {
        if (!table::contains(&storage.device_readings, device)) {
            return 0
        };
        vector::length(table::borrow(&storage.device_readings, device))
    }

    /// Get encrypted reading count
    public fun get_encrypted_reading_count(
        storage: &SensorStorage,
        device: address
    ): u64 {
        if (!table::contains(&storage.encrypted_readings, device)) {
            return 0
        };
        vector::length(table::borrow(&storage.encrypted_readings, device))
    }

    /// Get plaintext anomaly count
    public fun get_plaintext_anomaly_count(
        storage: &SensorStorage,
        device: address
    ): u64 {
        if (!table::contains(&storage.device_anomalies, device)) {
            return 0
        };
        vector::length(table::borrow(&storage.device_anomalies, device))
    }

    /// Get encrypted anomaly count
    public fun get_encrypted_anomaly_count(
        storage: &SensorStorage,
        device: address
    ): u64 {
        if (!table::contains(&storage.encrypted_anomalies, device)) {
            return 0
        };
        vector::length(table::borrow(&storage.encrypted_anomalies, device))
    }

    /// Get device metadata
    public fun get_device_metadata(
        storage: &SensorStorage,
        device_id: String
    ): (String, String, u64, bool) {
        if (!table::contains(&storage.device_metadata, device_id)) {
            return (string::utf8(b""), string::utf8(b""), 0, false)
        };
        
        let metadata = table::borrow(&storage.device_metadata, device_id);
        (metadata.device_name, metadata.location, metadata.total_readings, metadata.encryption_enabled)
    }

    /// Get total anomalies globally
    public fun get_total_anomalies(storage: &SensorStorage): u64 {
        storage.total_anomalies_detected
    }

    /// CSV hash functions
    public entry fun store_csv_hash(
        storage: &mut SensorStorage,
        file_hash: vector<u8>,
        timestamp: u64,
        _ctx: &mut TxContext
    ) {
        if (table::contains(&storage.csv_hashes, file_hash)) {
            let existing = table::borrow_mut(&mut storage.csv_hashes, file_hash);
            *existing = timestamp;
        } else {
            table::add(&mut storage.csv_hashes, file_hash, timestamp);
        };
    }

    public fun csv_hash_exists(
        storage: &SensorStorage,
        file_hash: vector<u8>
    ): bool {
        table::contains(&storage.csv_hashes, file_hash)
    }
}
