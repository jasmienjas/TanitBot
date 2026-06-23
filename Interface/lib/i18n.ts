export type Lang = "ar" | "en"

export type Topic = {
  id: string
  icon: "shield" | "lock" | "database" | "megaphone"
  title: string
  desc: string
  prompt: string
}

type Dict = {
  dir: "rtl" | "ltr"
  nav: { home: string; about: string; safety: string; getInvolved: string }
  badge: string
  heroTitle: string
  heroHighlight: string
  heroTitleEnd: string
  heroSubtitle: string
  startChat: string
  learnMore: string
  anonymous: string
  anonymousDesc: string
  chatTitle: string
  chatStatus: string
  inputPlaceholder: string
  send: string
  greeting: string
  suggestionsLabel: string
  topicsLabel: string
  topics: Topic[]
  disclaimer: string
  newChat: string
  thinking: string
  langLabel: string
}

export const t: Record<Lang, Dict> = {
  ar: {
    dir: "rtl",
    nav: { home: "الرئيسية", about: "من نحن", safety: "الأمان الرقمي", getInvolved: "انضم إلينا" },
    badge: "مساعد الأمان الرقمي بالذكاء الاصطناعي",
    heroTitle: "إرشادات فورية ومجهولة حول ",
    heroHighlight: "الأمان الرقمي",
    heroTitleEnd: " لكل التونسيين",
    heroSubtitle:
      "تانيت بوت مساعد ذكي يقدّم نصائح فورية ومجهولة الهوية حول الخصوصية وحماية البيانات وحرية التعبير، مع فهم للسياق التونسي ومصطلحاته المحلية.",
    startChat: "ابدأ المحادثة",
    learnMore: "اعرف المزيد",
    anonymous: "مجهول وآمن",
    anonymousDesc: "لا نطلب اسمك ولا نحفظ هويتك. تحدّث بحرية وأمان.",
    chatTitle: "تانيت بوت",
    chatStatus: "متصل الآن",
    inputPlaceholder: "اكتب سؤالك حول الأمان الرقمي...",
    send: "إرسال",
    greeting:
      "أهلا بيك! 👋 أنا تانيت بوت، نعاونك في كل ما يخص الأمان الرقمي والخصوصية وحماية بياناتك وحرية التعبير. اسألني أي سؤال، محادثتنا مجهولة تماماً.",
    suggestionsLabel: "جرّب تسأل:",
    topicsLabel: "مواضيع يمكنني مساعدتك فيها",
    topics: [
      {
        id: "account",
        icon: "shield",
        title: "حماية الحسابات",
        desc: "أمّن حساباتك على فيسبوك وإنستغرام وغيرها",
        prompt: "كيفاش نأمّن حساباتي على مواقع التواصل الاجتماعي؟",
      },
      {
        id: "privacy",
        icon: "lock",
        title: "الخصوصية",
        desc: "تحكّم في بياناتك الشخصية على الإنترنت",
        prompt: "كيفاش نحمي خصوصيتي على الإنترنت؟",
      },
      {
        id: "data",
        icon: "database",
        title: "حماية البيانات",
        desc: "حقوقك حسب القانون التونسي لحماية المعطيات",
        prompt: "شنوّة حقوقي في حماية بياناتي الشخصية في تونس؟",
      },
      {
        id: "expression",
        icon: "megaphone",
        title: "حرية التعبير",
        desc: "عبّر بأمان واعرف حدودك القانونية",
        prompt: "كيفاش نعبّر على رأيي على الإنترنت بأمان؟",
      },
    ],
    disclaimer:
      "تانيت بوت يقدّم إرشادات عامة ولا يُعتبر استشارة قانونية. للحالات الحرجة استشر مختصاً.",
    newChat: "محادثة جديدة",
    thinking: "تانيت بوت يكتب...",
    langLabel: "EN",
  },
  en: {
    dir: "ltr",
    nav: { home: "Home", about: "About", safety: "Digital Safety", getInvolved: "Get Involved" },
    badge: "AI-Powered Digital Safety Assistant",
    heroTitle: "Immediate, anonymous guidance on ",
    heroHighlight: "digital safety",
    heroTitleEnd: " for everyone in Tunisia",
    heroSubtitle:
      "TanitBot is an AI assistant offering instant, anonymous advice on privacy, data protection, and freedom of expression — grounded in the Tunisian context and local terminology.",
    startChat: "Start chatting",
    learnMore: "Learn more",
    anonymous: "Anonymous & Safe",
    anonymousDesc: "We never ask for your name or store your identity. Talk freely and securely.",
    chatTitle: "TanitBot",
    chatStatus: "Online now",
    inputPlaceholder: "Ask your digital safety question...",
    send: "Send",
    greeting:
      "Hi there! 👋 I'm TanitBot. I can help you with digital safety, privacy, protecting your data, and freedom of expression. Ask me anything — this conversation is completely anonymous.",
    suggestionsLabel: "Try asking:",
    topicsLabel: "Topics I can help with",
    topics: [
      {
        id: "account",
        icon: "shield",
        title: "Account Security",
        desc: "Secure your Facebook, Instagram and other accounts",
        prompt: "How do I secure my social media accounts?",
      },
      {
        id: "privacy",
        icon: "lock",
        title: "Privacy",
        desc: "Take control of your personal data online",
        prompt: "How can I protect my privacy online?",
      },
      {
        id: "data",
        icon: "database",
        title: "Data Protection",
        desc: "Your rights under Tunisian data protection law",
        prompt: "What are my personal data protection rights in Tunisia?",
      },
      {
        id: "expression",
        icon: "megaphone",
        title: "Freedom of Expression",
        desc: "Express yourself safely and know your legal limits",
        prompt: "How can I express my opinion online safely?",
      },
    ],
    disclaimer:
      "TanitBot provides general guidance and is not a substitute for legal advice. For critical situations, consult a professional.",
    newChat: "New chat",
    thinking: "TanitBot is typing...",
    langLabel: "ع",
  },
}
