import pandas as pd

print("⏳ Stanford (DeepChem) veri merkezine bağlanılıyor...\n")

# ---------------------------------------------------------
# 1. DOKU/ORGAN HASARI VERİ SETİ (Tox21)
# ---------------------------------------------------------
try:
    print("🫀 1. Doku/Organ Hasarı (Tox21) verisi indiriliyor...")
    url_tox = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    df_tox = pd.read_csv(url_tox, compression='gzip')

    # 12 farklı testten HERHANGİ BİRİNDE zehirli (1) çıktıysa, molekül 'Zehirli' kabul edilir
    toksisite_kolonlari = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    df_tox['toksik_mi'] = df_tox[toksisite_kolonlari].max(axis=1)
    df_tox = df_tox[['smiles', 'toksik_mi']].dropna()
    df_tox['toksik_mi'] = df_tox['toksik_mi'].astype(int)

    df_tox.to_csv('toksisite_veriseti.csv', index=False)
    print(f"✅ Başarılı! {len(df_tox)} adet Gerçek Toksisite verisi kaydedildi.\n")
except Exception as e:
    print(f"❌ Toksisite verisi indirilemedi: {e}\n")

# ---------------------------------------------------------
# 2. SİSTEMİK EMİLİM VERİ SETİ (Lipophilicity / ESOL Yedeği)
# ---------------------------------------------------------
try:
    print("📥 2. Sistemik Emilim verisi indiriliyor...")
    url_abs = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
    df_abs = pd.read_csv(url_abs)

    # LogD 1.0 ile 5.0 arası hücre zarından iyi emilir (1), diğerleri emilemez (0)
    df_abs['emilir_mi'] = df_abs['exp'].apply(lambda x: 1 if 1.0 <= x <= 5.0 else 0)
    df_abs = df_abs[['smiles', 'emilir_mi']].dropna()
    df_abs['emilir_mi'] = df_abs['emilir_mi'].astype(int)

    df_abs.to_csv('emilim_veriseti.csv', index=False)
    print(f"✅ Başarılı! {len(df_abs)} adet Gerçek Emilim verisi kaydedildi.\n")
except Exception as e:
    print(f"⚠️ İlk sunucu yanıt vermedi, alternatif emilim (ESOL) sunucusuna geçiliyor...")
    try:
        # ESOL: Suda çözünürlük veritabanı
        url_abs2 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        df_abs2 = pd.read_csv(url_abs2)

        # Suda çözünürlüğü makul seviyede olanlar (logS > -4.0) emilir (1)
        df_abs2['emilir_mi'] = df_abs2['measured log solubility in mols per litre'].apply(
            lambda x: 1 if x > -4.0 else 0)
        df_abs2 = df_abs2[['smiles', 'emilir_mi']].dropna()
        df_abs2['emilir_mi'] = df_abs2['emilir_mi'].astype(int)

        df_abs2.to_csv('emilim_veriseti.csv', index=False)
        print(f"✅ Başarılı! {len(df_abs2)} adet Alternatif Emilim verisi kaydedildi.\n")
    except Exception as ex:
        print(f"❌ Emilim verisi indirilemedi: {ex}\n")

# ---------------------------------------------------------
# 3. ALZHEIMER HEDEF ETKİNLİĞİ VERİ SETİ (BACE-1 İnhibisyonu)
# ---------------------------------------------------------
try:
    print("🎯 3. Alzheimer Etkinlik (BACE-1 İnhibisyonu) verisi indiriliyor...")
    url_eff = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
    df_eff = pd.read_csv(url_eff)

    # Class = 1 (Alzheimer enzimi durduruldu), 0 (İşe yaramadı)
    df_eff = df_eff.rename(columns={'mol': 'smiles', 'Class': 'etkili_mi'})
    df_eff = df_eff[['smiles', 'etkili_mi']].dropna()

    df_eff.to_csv('etkinlik_veriseti.csv', index=False)
    print(f"✅ Başarılı! {len(df_eff)} adet Gerçek Alzheimer Etkinlik verisi kaydedildi.\n")
except Exception as e:
    print(f"❌ Alzheimer Etkinlik verisi indirilemedi: {e}\n")

print("🎉 HARİKA! Tüm gerçek veri setleri hazır. Artık simülatörü başlatabilirsiniz!")